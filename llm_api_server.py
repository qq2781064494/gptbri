#!/usr/bin/env python3
"""OpenAI-compatible local API server backed by the ChatGPT web bridge."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextvars
import contextlib
import json
import logging
import mimetypes
import os
import re
import secrets
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from chatgpt_bridge import (
    DEFAULT_PROFILE_DIR,
    ChatGPTBridgeError,
    ChatGPTWebBridge,
)


DEFAULT_CDP_URL = "http://127.0.0.1:9222"
DEFAULT_API_CHAT_URL = "https://chatgpt.com/"
DEFAULT_MODEL_NAME = "chatgpt-web"
DEFAULT_CHAT_CONVERSATION_KEY = "chat"
DEFAULT_TOOL_CONVERSATION_KEY = "tool"
LOG_PREVIEW_CHARS = 10
LOG_PREVIEW_BYTES = 8192
LOG_PREVIEW_CAPTURE_BYTES = 256 * 1024


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def setup_logging() -> logging.Logger:
    level_name = (env_str("BRIDGE_LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("chatgpt_bridge_api")
    logger.setLevel(level)
    return logger


logger = setup_logging()
CURRENT_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chatgpt_bridge_request_id",
    default="-",
)


def current_request_id() -> str:
    return CURRENT_REQUEST_ID.get("-")


def detailed_logs_enabled() -> bool:
    return env_bool("BRIDGE_DETAILED_LOGS", False)


def detailed_log(message: str, *args: Any) -> None:
    if detailed_logs_enabled():
        logger.info(message, *args)


def model_name() -> str:
    return env_str("BRIDGE_MODEL_NAME", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME


class Runtime:
    def __init__(self) -> None:
        self.bridge: ChatGPTWebBridge | None = None
        self.locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, conversation_key: str) -> asyncio.Lock:
        key = conversation_key or "default"
        lock = self.locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[key] = lock
        return lock


runtime = Runtime()


class RequestTimingLogMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_from_scope(scope) or uuid.uuid4().hex[:12]
        method = str(scope.get("method") or "")
        path = str(scope.get("path") or "")
        client = scope.get("client")
        client_addr = client_address(scope)
        started_at = time.time()
        status_code: int | None = None
        returned = False
        request_messages = await read_request_messages(receive)
        request_preview = body_preview_from_messages(request_messages)
        response_preview = BodyPreviewCapture()
        receive_replay = replay_request_messages(request_messages)
        request_id_token = CURRENT_REQUEST_ID.set(request_id)

        logger.info(
            (
                "request.received id=%s method=%s path=%s client=%s "
                "request_bytes=%s request_text_head=%r request_text_tail=%r"
            ),
            request_id,
            method,
            path,
            client_addr,
            request_preview["bytes"],
            request_preview["head"],
            request_preview["tail"],
        )

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code, returned
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 0)
                headers = list(message.get("headers") or [])
                headers.append((b"x-request-id", request_id.encode("latin-1", "ignore")))
                message = {**message, "headers": headers}
            elif (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_preview.add(message.get("body") or b"")
                returned = True
                log_request_returned(
                    request_id,
                    method,
                    path,
                    status_code,
                    started_at,
                    request_preview,
                    response_preview.preview(),
                )
            elif message.get("type") == "http.response.body":
                response_preview.add(message.get("body") or b"")

            await send(message)

        try:
            await self.app(scope, receive_replay, send_wrapper)
        except Exception:
            duration_ms = (time.time() - started_at) * 1000
            response = response_preview.preview()
            logger.exception(
                (
                    "request.failed id=%s method=%s path=%s status=%s duration_ms=%.1f "
                    "request_bytes=%s request_text_head=%r request_text_tail=%r "
                    "response_bytes=%s response_text_head=%r response_text_tail=%r"
                ),
                request_id,
                method,
                path,
                status_code or "-",
                duration_ms,
                request_preview["bytes"],
                request_preview["head"],
                request_preview["tail"],
                response["bytes"],
                response["head"],
                response["tail"],
            )
            raise
        finally:
            if not returned and status_code is not None:
                log_request_returned(
                    request_id,
                    method,
                    path,
                    status_code,
                    started_at,
                    request_preview,
                    response_preview.preview(),
                )
            CURRENT_REQUEST_ID.reset(request_id_token)


class BodyPreviewCapture:
    def __init__(self) -> None:
        self.total_bytes = 0
        self.head = bytearray()
        self.tail = bytearray()
        self.sample = bytearray()

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return

        self.total_bytes += len(chunk)
        if len(self.sample) < LOG_PREVIEW_CAPTURE_BYTES:
            needed = LOG_PREVIEW_CAPTURE_BYTES - len(self.sample)
            self.sample.extend(chunk[:needed])

        if len(self.head) < LOG_PREVIEW_BYTES:
            needed = LOG_PREVIEW_BYTES - len(self.head)
            self.head.extend(chunk[:needed])

        self.tail.extend(chunk)
        if len(self.tail) > LOG_PREVIEW_BYTES:
            del self.tail[: len(self.tail) - LOG_PREVIEW_BYTES]

    def preview(self) -> dict[str, Any]:
        semantic_text = extract_log_semantic_text(bytes(self.sample))
        if semantic_text:
            return text_preview(semantic_text, self.total_bytes)
        return body_preview_from_parts(bytes(self.head), bytes(self.tail), self.total_bytes)


async def read_request_messages(receive: Any) -> list[dict[str, Any]]:
    messages = []
    while True:
        message = await receive()
        messages.append(message)
        if message.get("type") != "http.request":
            break
        if not message.get("more_body", False):
            break
    return messages


def replay_request_messages(messages: list[dict[str, Any]]) -> Any:
    index = 0
    wait_forever = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal index
        if index < len(messages):
            message = messages[index]
            index += 1
            return message

        # StreamingResponse keeps a background disconnect listener that awaits
        # receive(). Returning empty http.request messages forever creates a
        # tight loop and prevents the stream task from sending chunks.
        await wait_forever.wait()
        return {"type": "http.disconnect"}

    return receive


def body_preview_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [
        message.get("body") or b""
        for message in messages
        if message.get("type") == "http.request"
    ]
    body = b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))
    return body_preview_from_bytes(body)


def body_preview_from_bytes(body: bytes, total_bytes: int | None = None) -> dict[str, Any]:
    byte_count = len(body) if total_bytes is None else total_bytes
    semantic_text = extract_log_semantic_text(body)
    if semantic_text:
        return text_preview(semantic_text, byte_count)

    return body_preview_from_parts(body[:LOG_PREVIEW_BYTES], body[-LOG_PREVIEW_BYTES:], byte_count)


def text_preview(text: str, total_bytes: int) -> dict[str, Any]:
    normalized = normalize_log_text(text)
    return {
        "bytes": total_bytes,
        "head": normalized[:LOG_PREVIEW_CHARS],
        "tail": normalized[-LOG_PREVIEW_CHARS:],
    }


def body_preview_from_parts(
    head_bytes: bytes,
    tail_bytes: bytes,
    total_bytes: int,
) -> dict[str, Any]:
    head_text = decode_log_preview(head_bytes)
    tail_text = decode_log_preview(tail_bytes)
    return {
        "bytes": total_bytes,
        "head": head_text[:LOG_PREVIEW_CHARS] if head_text else "",
        "tail": tail_text[-LOG_PREVIEW_CHARS:] if tail_text else "",
    }


def extract_log_semantic_text(body: bytes) -> str:
    if not body:
        return ""

    text = body.decode("utf-8", errors="replace")
    payload = parse_log_json(text)
    if payload is not None:
        return extract_json_payload_text(payload)

    if "data:" in text:
        return extract_sse_payload_text(text)

    return ""


def parse_log_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def extract_sse_payload_text(text: str) -> str:
    parts = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        payload = parse_log_json(data)
        if payload is None:
            parts.append(data)
        else:
            extracted = extract_json_payload_text(payload)
            if extracted:
                parts.append(extracted)
    return "\n".join(parts)


def extract_json_payload_text(payload: Any) -> str:
    parts: list[str] = []
    if isinstance(payload, dict):
        instructions = payload.get("instructions")
        if isinstance(instructions, str):
            parts.append(instructions)

        for key in ("messages", "input", "output"):
            if key in payload:
                parts.extend(extract_payload_text_parts(payload[key]))

        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                parts.extend(extract_payload_text_parts(choice.get("message")))
                parts.extend(extract_payload_text_parts(choice.get("delta")))

        output_text = payload.get("output_text")
        if isinstance(output_text, str):
            parts.append(output_text)

        if not parts:
            parts.extend(extract_payload_text_parts(payload))
    else:
        parts.extend(extract_payload_text_parts(payload))

    return "\n".join(part for part in parts if part)


def extract_payload_text_parts(value: Any) -> list[str]:
    parts: list[str] = []
    if value is None:
        return parts
    if isinstance(value, str):
        if value:
            parts.append(value)
        return parts
    if isinstance(value, (int, float, bool)):
        parts.append(str(value))
        return parts
    if isinstance(value, list):
        for item in value:
            parts.extend(extract_payload_text_parts(item))
        return parts
    if not isinstance(value, dict):
        return parts

    value_type = str(value.get("type") or "")
    role = str(value.get("role") or "")
    if isinstance(value.get("content"), str):
        parts.append(with_log_role(role, value["content"]))
    else:
        parts.extend(extract_payload_text_parts(value.get("content")))

    for key in ("text", "output_text"):
        item = value.get(key)
        if isinstance(item, str):
            parts.append(with_log_role(role, item))

    image_url = value.get("image_url") or value.get("input_image")
    if image_url:
        parts.append("[image]")

    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
        if name or arguments:
            parts.append(f"{name or 'function'}({arguments or ''})")

    if value_type in {"function_call", "tool_call"}:
        name = value.get("name") or value.get("call_id") or value_type
        arguments = value.get("arguments") or value.get("input") or ""
        parts.append(f"{name}({arguments})")

    for key in ("message", "delta", "output", "input"):
        if key in value:
            parts.extend(extract_payload_text_parts(value[key]))

    return parts


def with_log_role(role: str, text: str) -> str:
    return text


def normalize_log_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def decode_log_preview(data: bytes) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    return normalize_log_text(text)


def request_id_from_scope(scope: dict[str, Any]) -> str | None:
    for name, value in scope.get("headers") or []:
        if name.lower() == b"x-request-id":
            decoded = value.decode("latin-1").strip()
            return decoded or None
    return None


def client_address(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if isinstance(client, tuple) and len(client) >= 2:
        return f"{client[0]}:{client[1]}"
    if isinstance(client, tuple) and client:
        return str(client[0])
    return "-"


def log_request_returned(
    request_id: str,
    method: str,
    path: str,
    status_code: int | None,
    started_at: float,
    request_preview: dict[str, Any],
    response_preview: dict[str, Any],
) -> None:
    duration_ms = (time.time() - started_at) * 1000
    logger.info(
        (
            "request.returned id=%s method=%s path=%s status=%s duration_ms=%.1f "
            "request_bytes=%s request_text_head=%r request_text_tail=%r "
            "response_bytes=%s response_text_head=%r response_text_tail=%r"
        ),
        request_id,
        method,
        path,
        status_code or "-",
        duration_ms,
        request_preview["bytes"],
        request_preview["head"],
        request_preview["tail"],
        response_preview["bytes"],
        response_preview["head"],
        response_preview["tail"],
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    bridge = ChatGPTWebBridge(
        chat_url=env_str("BRIDGE_CHAT_URL", DEFAULT_API_CHAT_URL) or DEFAULT_API_CHAT_URL,
        profile_dir=env_str("BRIDGE_PROFILE_DIR", DEFAULT_PROFILE_DIR) or DEFAULT_PROFILE_DIR,
        browser_channel=env_str("BRIDGE_BROWSER_CHANNEL"),
        cdp_url=env_str("BRIDGE_CDP_URL", DEFAULT_CDP_URL),
        reuse_cdp_page=env_bool("BRIDGE_REUSE_CDP_PAGE", True),
        close_extra_chatgpt_pages=env_bool("BRIDGE_CLOSE_EXTRA_CHATGPT_PAGES", True),
        chat_reset_seconds=env_int("BRIDGE_CHAT_RESET_SECONDS", 0),
        headless=env_bool("BRIDGE_HEADLESS", False),
        timeout_ms=env_int("BRIDGE_TIMEOUT_SECONDS", 180) * 1000,
    )
    await bridge.__aenter__()
    if env_bool("BRIDGE_PREOPEN_CONVERSATION_PAGES", True):
        await bridge.ensure_conversation_pages(
            [
                env_str("BRIDGE_CHAT_CONVERSATION_KEY", DEFAULT_CHAT_CONVERSATION_KEY)
                or DEFAULT_CHAT_CONVERSATION_KEY,
                env_str("BRIDGE_TOOL_CONVERSATION_KEY", DEFAULT_TOOL_CONVERSATION_KEY)
                or DEFAULT_TOOL_CONVERSATION_KEY,
            ]
        )
    runtime.bridge = bridge
    try:
        yield
    finally:
        runtime.bridge = None
        await bridge.__aexit__(None, None, None)


app = FastAPI(title="ChatGPT Web Bridge API", version="1.0.0", lifespan=lifespan)
app.add_middleware(RequestTimingLogMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "code": None,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "code": None,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    message = str(exc) or exc.__class__.__name__
    if env_bool("BRIDGE_DEBUG_ERRORS", True):
        message = f"{message}\n{traceback.format_exc()}"

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": message,
                "type": "server_error",
                "code": None,
            }
        },
    )


async def require_api_key(request: Request) -> None:
    expected = env_str("BRIDGE_API_KEY")
    if not expected:
        return

    auth_header = request.headers.get("authorization", "")
    bearer_prefix = "Bearer "
    token = ""
    if auth_header.startswith(bearer_prefix):
        token = auth_header[len(bearer_prefix) :].strip()
    token = token or request.headers.get("x-api-key", "").strip()

    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime.bridge else "starting",
        "model": model_name(),
        "cdp_url": env_str("BRIDGE_CDP_URL", DEFAULT_CDP_URL),
        "chat_reset_seconds": env_int("BRIDGE_CHAT_RESET_SECONDS", 0),
        "chat_conversation_key": env_str(
            "BRIDGE_CHAT_CONVERSATION_KEY",
            DEFAULT_CHAT_CONVERSATION_KEY,
        ),
        "tool_conversation_key": env_str(
            "BRIDGE_TOOL_CONVERSATION_KEY",
            DEFAULT_TOOL_CONVERSATION_KEY,
        ),
    }


@app.get("/models", dependencies=[Depends(require_api_key)])
@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_name(),
                "object": "model",
                "created": 0,
                "owned_by": "chatgpt-web-bridge",
            }
        ],
    }


@app.post("/chat/completions", dependencies=[Depends(require_api_key)], response_model=None)
@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)], response_model=None)
async def create_chat_completion(payload: dict[str, Any]) -> JSONResponse | StreamingResponse:
    request_model = str(payload.get("model") or model_name())
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise openai_error(422, "messages must be a non-empty array.")

    detailed_log(
        "chat.route.start id=%s model=%s stream=%s message_count=%s",
        current_request_id(),
        request_model,
        payload.get("stream") is True,
        len(messages),
    )
    result = await complete_chat_request(payload, messages, request_model)
    detailed_log(
        "chat.route.result id=%s kind=%s prompt_chars=%s",
        current_request_id(),
        result.get("kind"),
        len(str(result.get("prompt") or "")),
    )
    if result["kind"] == "tool_calls":
        response_payload = tool_call_response(
            result["tool_calls"],
            request_model,
            result["prompt"],
        )
        detailed_log(
            "chat.route.return_json id=%s kind=tool_calls response_bytes=%s",
            current_request_id(),
            len(json.dumps(response_payload, ensure_ascii=False).encode("utf-8")),
        )
        return JSONResponse(response_payload)

    reply = str(result["reply"])
    if payload.get("stream") is True:
        detailed_log(
            "chat.route.return_stream id=%s reply_chars=%s",
            current_request_id(),
            len(reply),
        )
        return StreamingResponse(
            stream_response(reply, request_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    response_payload = chat_completion_response(reply, request_model, result["prompt"])
    detailed_log(
        "chat.route.return_json id=%s kind=message reply_chars=%s response_bytes=%s",
        current_request_id(),
        len(reply),
        len(json.dumps(response_payload, ensure_ascii=False).encode("utf-8")),
    )
    return JSONResponse(response_payload)


@app.post("/responses", dependencies=[Depends(require_api_key)], response_model=None)
@app.post("/v1/responses", dependencies=[Depends(require_api_key)], response_model=None)
async def create_response(payload: dict[str, Any]) -> JSONResponse | StreamingResponse:
    request_model = str(payload.get("model") or model_name())
    messages = responses_input_to_messages(payload)
    if not messages:
        raise openai_error(422, "input must contain at least one message.")

    result = await complete_chat_request(payload, messages, request_model)
    response_payload = responses_api_response(result, request_model)
    if payload.get("stream") is True:
        return StreamingResponse(
            stream_responses_response(response_payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(response_payload)


async def complete_chat_request(
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    request_model: str,
) -> dict[str, Any]:
    if runtime.bridge is None:
        raise openai_error(503, "Bridge is not ready.")

    tools = extract_tools(payload)
    tool_choice = payload.get("tool_choice", "auto")
    has_tools_result = has_tool_result(messages)
    conversation_key = request_conversation_key(tools, has_tools_result)
    force_new_chat = should_force_new_chat(messages, tools, has_tools_result)
    detailed_log(
        (
            "chat.complete.start id=%s model=%s key=%s tools=%s has_tools_result=%s "
            "tool_choice=%r force_new=%s"
        ),
        current_request_id(),
        request_model,
        conversation_key,
        len(tools),
        has_tools_result,
        tool_choice,
        force_new_chat,
    )
    if tools and tool_choice != "none" and not has_tools_result:
        with tempfile.TemporaryDirectory(prefix="chatgpt_bridge_api_") as tmp_dir:
            prompt, images = await build_prompt_and_images(messages, Path(tmp_dir))
            detailed_log(
                "chat.complete.plan_prompt id=%s key=%s prompt_chars=%s images=%s",
                current_request_id(),
                conversation_key,
                len(prompt),
                len(images),
            )
            tool_calls, final_answer = await choose_tool_response(
                messages,
                tools,
                tool_choice,
                conversation_key,
                force_new_chat,
                prompt,
                images,
            )
        if tool_calls:
            return {
                "kind": "tool_calls",
                "tool_calls": tool_calls,
                "prompt": prompt,
            }
        if final_answer:
            detailed_log(
                "chat.complete.plan_final_answer id=%s key=%s reply_chars=%s",
                current_request_id(),
                conversation_key,
                len(final_answer),
            )
            return {
                "kind": "message",
                "reply": final_answer,
                "prompt": prompt,
            }

    with tempfile.TemporaryDirectory(prefix="chatgpt_bridge_api_") as tmp_dir:
        if has_tools_result:
            prompt, images = await build_final_tool_answer_prompt(messages, Path(tmp_dir))
        else:
            prompt, images = await build_prompt_and_images(messages, Path(tmp_dir))
        detailed_log(
            "chat.complete.prompt_ready id=%s key=%s prompt_chars=%s images=%s",
            current_request_id(),
            conversation_key,
            len(prompt),
            len(images),
        )

        async with runtime.lock_for(conversation_key):
            try:
                detailed_log(
                    "chat.complete.bridge_call id=%s key=%s prompt_chars=%s images=%s",
                    current_request_id(),
                    conversation_key,
                    len(prompt),
                    len(images),
                )
                reply = await runtime.bridge.ask(
                    text=isolate_reused_chat_prompt(prompt),
                    images=images,
                    conversation_key=conversation_key,
                    force_new_chat=force_new_chat,
                    log_label=current_request_id(),
                )
                detailed_log(
                    "chat.complete.bridge_returned id=%s key=%s reply_chars=%s",
                    current_request_id(),
                    conversation_key,
                    len(reply),
                )
            except ChatGPTBridgeError as exc:
                raise openai_error(502, str(exc)) from exc

    unwrapped_reply = unwrap_planner_final_answer(reply)
    detailed_log(
        "chat.complete.return_message id=%s key=%s reply_chars=%s unwrapped_chars=%s",
        current_request_id(),
        conversation_key,
        len(reply),
        len(unwrapped_reply),
    )
    return {
        "kind": "message",
        "reply": unwrapped_reply,
        "prompt": prompt,
    }


def request_conversation_key(tools: list[dict[str, Any]], has_tools_result: bool) -> str:
    if tools or has_tools_result:
        return (
            env_str("BRIDGE_TOOL_CONVERSATION_KEY", DEFAULT_TOOL_CONVERSATION_KEY)
            or DEFAULT_TOOL_CONVERSATION_KEY
        )
    return (
        env_str("BRIDGE_CHAT_CONVERSATION_KEY", DEFAULT_CHAT_CONVERSATION_KEY)
        or DEFAULT_CHAT_CONVERSATION_KEY
    )


def should_force_new_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    has_tools_result: bool,
) -> bool:
    if env_bool("BRIDGE_FORCE_NEW_CHAT_PER_REQUEST", False):
        return True
    if tools or has_tools_result:
        return env_bool("BRIDGE_FORCE_NEW_TOOL_CHAT", False)
    if looks_like_codex_request(messages):
        return env_bool("BRIDGE_FORCE_NEW_CODEX_CHAT", False)
    return False


def isolate_reused_chat_prompt(prompt: str) -> str:
    if not env_bool("BRIDGE_ISOLATE_REUSED_CHAT", True):
        return prompt

    request_id = uuid.uuid4().hex[:12]
    return (
        f"【独立请求 {request_id}】\n"
        "请忽略本 ChatGPT 网页对话中此前的所有消息、项目背景、工具规划 JSON 和回答风格。\n"
        "只依据下面这一次 API 请求中的内容作答；不要输出本段边界、请求 ID 或任何内部协议。\n\n"
        f"{prompt}"
    )


def looks_like_codex_request(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        if role not in {"system", "developer"}:
            continue
        text = content_to_text(message.get("content", ""))
        if "You are Codex" in text or "coding agent based on" in text:
            return True
    return False


def responses_input_to_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    raw_input = payload.get("input")
    if raw_input is None:
        raw_input = payload.get("messages")

    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            messages.extend(response_input_item_to_messages(item))
    elif isinstance(raw_input, dict):
        messages.extend(response_input_item_to_messages(raw_input))
    elif raw_input is not None:
        messages.append({"role": "user", "content": str(raw_input)})

    return messages


def response_input_item_to_messages(item: Any) -> list[dict[str, Any]]:
    if isinstance(item, str):
        return [{"role": "user", "content": item}]
    if not isinstance(item, dict):
        return [{"role": "user", "content": str(item)}]

    item_type = str(item.get("type") or "")
    role = str(item.get("role") or "")
    if item_type == "message" or role in {"system", "developer", "user", "assistant"}:
        message_role = role or "user"
        return [
            {
                "role": message_role,
                "content": responses_content_to_chat(item.get("content", "")),
            }
        ]

    if item_type == "function_call":
        arguments = item.get("arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}")
        return [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": str(item.get("name") or "unknown"),
                            "arguments": arguments,
                        },
                    }
                ],
            }
        ]

    if item_type == "function_call_output":
        return [
            {
                "role": "tool",
                "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                "content": response_output_to_text(item.get("output", "")),
            }
        ]

    if item_type in {"input_text", "text"}:
        return [{"role": "user", "content": str(item.get("text") or "")}]
    if item_type == "output_text":
        return [{"role": "assistant", "content": str(item.get("text") or "")}]
    if item_type in {"input_image", "image_url"}:
        return [{"role": "user", "content": [item]}]
    if item_type:
        return [
            {
                "role": role if role in {"system", "developer", "user", "assistant"} else "user",
                "content": json.dumps(item, ensure_ascii=False),
            }
        ]

    return [{"role": "user", "content": json.dumps(item, ensure_ascii=False)}]


def responses_content_to_chat(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    converted: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue

        part_type = part.get("type")
        if part_type in {"input_text", "output_text"}:
            converted.append({"type": "text", "text": str(part.get("text") or "")})
            continue
        converted.append(part)

    return converted


def response_output_to_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def extract_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = payload.get("tools") or []
    functions = payload.get("functions") or []

    tools: list[dict[str, Any]] = []
    if isinstance(raw_tools, list):
        for tool_spec in raw_tools:
            if not isinstance(tool_spec, dict):
                continue
            if tool_spec.get("type") == "function" and isinstance(tool_spec.get("function"), dict):
                function = tool_spec["function"]
            else:
                function = tool_spec
            name = function.get("name")
            if name:
                tools.append(
                    {
                        "name": str(name),
                        "description": str(function.get("description") or ""),
                        "parameters": function.get("parameters") or {},
                    }
                )

    if isinstance(functions, list):
        for function in functions:
            if isinstance(function, dict) and function.get("name"):
                tools.append(
                    {
                        "name": str(function["name"]),
                        "description": str(function.get("description") or ""),
                        "parameters": function.get("parameters") or {},
                    }
                )

    return tools


def has_tool_result(messages: list[dict[str, Any]]) -> bool:
    return any(str(message.get("role") or "") in {"tool", "function"} for message in messages)


async def choose_tool_response(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any,
    conversation_key: str,
    force_new_chat: bool = False,
    request_prompt: str | None = None,
    request_images: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    forced_name = forced_tool_name(tool_choice)
    should_require_tool = should_require_tool_call(messages, tools, forced_name)
    planned = await plan_tool_calls_with_web(
        messages,
        tools,
        forced_name,
        conversation_key,
        force_new_chat,
        request_prompt,
        request_images,
    )
    if planned[0]:
        return planned[0], None

    deterministic = deterministic_tool_calls(messages, tools, forced_name)
    if deterministic and (
        should_require_tool or env_bool("BRIDGE_USE_LOCAL_TOOL_FALLBACK", False)
    ):
        return deterministic, None

    if planned[1] and not should_require_tool:
        return [], planned[1]

    return [], None


def forced_tool_name(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    if tool_choice.get("type") == "function" and tool_choice.get("name"):
        return str(tool_choice["name"])
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return None


def should_require_tool_call(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    forced_name: str | None = None,
) -> bool:
    if forced_name:
        return True
    return needs_workspace_inspection(messages) and tool_by_name(tools, "exec_command") is not None


def deterministic_tool_calls(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    forced_name: str | None = None,
) -> list[dict[str, Any]]:
    query = latest_user_text(messages)
    if not query:
        return []

    available = {tool["name"].lower(): tool for tool in tools}

    workspace_calls = deterministic_workspace_tool_calls(messages, tools, forced_name)
    if workspace_calls:
        return workspace_calls

    lower_query = query.lower()
    numbers = [int(value) for value in re.findall(r"-?\d+", query)]

    selected_name = forced_name
    if not selected_name:
        if "multiply" in available and re.search(r"(\*|×|x|乘以?|相乘|product)", lower_query):
            selected_name = "multiply"
        elif "add" in available and re.search(r"(\+|加|相加|求和|之和|sum)", lower_query):
            selected_name = "add"

    if not selected_name:
        return []

    tool = available.get(selected_name.lower())
    if not tool:
        return []

    arguments = infer_tool_arguments(tool, numbers)
    return [make_tool_call(tool["name"], arguments)]


def deterministic_workspace_tool_calls(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    forced_name: str | None = None,
) -> list[dict[str, Any]]:
    if not needs_workspace_inspection(messages):
        return []

    exec_tool = tool_by_name(tools, forced_name or "exec_command")
    if not exec_tool or exec_tool["name"].lower() != "exec_command":
        return []

    return [
        make_tool_call(
            exec_tool["name"],
            sanitize_tool_arguments(
                exec_tool,
                {
                    "cmd": "rg --files",
                    "yield_time_ms": 10000,
                    "max_output_tokens": 20000,
                },
            ),
        ),
        make_tool_call(
            exec_tool["name"],
            sanitize_tool_arguments(
                exec_tool,
                {
                    "cmd": "sed -n '1,260p' README.md",
                    "yield_time_ms": 10000,
                    "max_output_tokens": 24000,
                },
            ),
        ),
    ]


def needs_workspace_inspection(messages: list[dict[str, Any]]) -> bool:
    query = latest_user_text(messages)
    if not query or is_prompt_transformation_request(query):
        return False

    lower_query = query.lower()
    workspace_pattern = re.compile(
        r"(这个|当前|本地)?(项目|工程|仓库|代码库|目录|文件)"
        r"|workspace|codebase|repo(?:sitory)?|project|current directory"
    )
    inspect_pattern = re.compile(
        r"看看|看一下|看下|读一下|读下|说了什么|讲了什么|做什么|干什么|"
        r"总结|概括|介绍|分析|检查|找|在哪|定位|"
        r"look at|inspect|read|summari[sz]e|analy[sz]e|find|locate|where"
    )
    return bool(workspace_pattern.search(lower_query) and inspect_pattern.search(lower_query))


def is_prompt_transformation_request(text: str) -> bool:
    lower_text = text.lower()
    markers = [
        "provide a short title",
        "generate a concise ui title",
        "do not respond to the user",
        "just write a title",
        "user prompt:",
        "title field",
        "不要解决",
        "只写标题",
        "生成标题",
    ]
    return any(marker in lower_text for marker in markers)


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "") == "user":
            return content_to_text(message.get("content", ""))
    return ""


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def infer_tool_arguments(tool: dict[str, Any], numbers: list[int]) -> dict[str, Any]:
    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    names = list(properties.keys()) if isinstance(properties, dict) else []
    if not names:
        names = ["a", "b"] if len(numbers) >= 2 else []

    arguments: dict[str, Any] = {}
    for index, name in enumerate(names):
        if index < len(numbers):
            arguments[name] = numbers[index]
    return arguments


def tool_by_name(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    lower_name = name.lower()
    for tool in tools:
        if str(tool.get("name") or "").lower() == lower_name:
            return tool
    return None


def sanitize_tool_arguments(tool: dict[str, Any], arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return {}

    parameters = tool.get("parameters") or {}
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict):
        return dict(arguments)

    normalized = dict(arguments)
    if "cmd" in properties and "cmd" not in normalized and "command" in normalized:
        normalized["cmd"] = normalized["command"]

    return {
        str(name): normalized[name]
        for name in properties
        if name in normalized
    }


def make_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def normalize_tool_call(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None

    call_id = str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}")
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        raw_arguments = function.get("arguments") or {}
    else:
        name = tool_call.get("name")
        raw_arguments = tool_call.get("args") or tool_call.get("arguments") or {}

    if not name:
        return None

    arguments: dict[str, Any]
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            parsed = {}
        arguments = parsed if isinstance(parsed, dict) else {}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {}

    return {"id": call_id, "name": str(name), "arguments": arguments}


async def plan_tool_calls_with_web(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    forced_name: str | None = None,
    conversation_key: str = DEFAULT_TOOL_CONVERSATION_KEY,
    force_new_chat: bool = False,
    request_prompt: str | None = None,
    request_images: list[Path] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    if runtime.bridge is None:
        return [], None

    full_request = (request_prompt or "").strip() or latest_user_text(messages)
    planner_prompt = (
        "你是一个 OpenAI tool calling 规划器。你的任务是根据用户问题和工具定义，"
        "判断是否需要调用工具，并输出严格 JSON。\n\n"
        "工具调用判断原则：\n"
        "- 先判断当前用户问题是否缺少回答所需证据，而不是先寻找直接回答的理由。\n"
        "- 只要缺少当前项目、代码库、文件、目录、运行结果、系统状态、网页、图片、时间"
        "或其他当前/外部事实，并且可用工具能够获取或验证这些事实，就必须调用最相关的工具。\n"
        "- 工具调用的目标是补齐缺失证据：优先选择能直接读取、搜索、执行、观察或验证"
        "缺失信息的工具。\n"
        "- 不要要求用户提供已经能通过工具获取的信息；不要凭空猜测；不要用“我无法查看”"
        "替代工具调用。\n"
        "- 只有当前消息已经提供足够证据，或者任务属于闲聊、纯常识、翻译、改写、标题生成、"
        "摘要用户已粘贴内容等不需要外部证据的情况，才可以不调用工具直接回答。\n\n"
        "硬性要求：\n"
        "1. 只输出一个 JSON 对象，不要 Markdown，不要代码块，不要解释。\n"
        "2. 如果需要调用工具，final_answer 必须是 null。\n"
        "3. 如果不需要调用工具，tool_calls 必须是空数组。\n"
        "4. arguments 必须是对象，参数名必须来自工具 parameters.properties。\n\n"
        "输出格式：\n"
        '{"tool_calls":[{"name":"工具名","arguments":{"参数名":参数值}}],"final_answer":null}\n'
        "或：\n"
        '{"tool_calls":[],"final_answer":"直接回答"}\n\n'
        f"强制工具名：{forced_name or '无'}\n\n"
        f"可用工具 JSON：\n{json.dumps(tools, ensure_ascii=False)}\n\n"
        f"完整请求消息上下文：\n{full_request}"
    )

    async with runtime.lock_for(conversation_key):
        try:
            reply = await runtime.bridge.ask(
                text=isolate_reused_chat_prompt(planner_prompt),
                images=request_images or [],
                conversation_key=conversation_key,
                force_new_chat=force_new_chat,
                log_label=current_request_id(),
            )
        except ChatGPTBridgeError:
            return [], None

    payload = parse_json_object(reply)
    if not payload:
        return [], None

    calls = []
    for item in payload.get("tool_calls") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        calls.append(make_tool_call(str(item["name"]), arguments))

    final_answer = payload.get("final_answer")
    return calls, str(final_answer) if final_answer else None


def parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def unwrap_planner_final_answer(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    if not (stripped.startswith("{") and stripped.endswith("}")):
        return text

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return text

    if not isinstance(payload, dict):
        return text
    if set(payload).issubset({"tool_calls", "final_answer"}) and not payload.get("tool_calls"):
        final_answer = payload.get("final_answer")
        if isinstance(final_answer, str) and final_answer:
            return final_answer
    return text


def openai_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": None,
            }
        },
    )


async def build_prompt_and_images(
    messages: list[dict[str, Any]],
    tmp_dir: Path,
) -> tuple[str, list[Path]]:
    blocks: list[str] = []
    images: list[Path] = []

    for message in messages:
        if not isinstance(message, dict):
            raise openai_error(422, "Each message must be an object.")

        role = str(message.get("role") or "user")
        content = message.get("content", "")
        text_chunks, message_images = await parse_message_content(content, tmp_dir)
        images.extend(message_images)

        text = "\n".join(chunk for chunk in text_chunks if chunk.strip()).strip()
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            text = "\n".join(
                part
                for part in [
                    text,
                    "工具调用请求：",
                    json.dumps(tool_calls, ensure_ascii=False),
                ]
                if part
            )
        if role in {"tool", "function"}:
            tool_name = message.get("name") or message.get("tool_call_id") or "unknown"
            text = f"{tool_name} 返回：{text}"

        if text:
            blocks.append(f"{role.upper()}:\n{text}")

    prompt = "\n\n".join(blocks).strip()
    if not prompt:
        prompt = "请根据上传的图片回答。"
    return prompt, images


async def build_final_tool_answer_prompt(
    messages: list[dict[str, Any]],
    tmp_dir: Path,
) -> tuple[str, list[Path]]:
    message_blocks: list[str] = []
    images: list[Path] = []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    tool_results: list[dict[str, str]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "")
        content = message.get("content", "")
        text_chunks, message_images = await parse_message_content(content, tmp_dir)
        text = "\n".join(chunk for chunk in text_chunks if chunk.strip()).strip()
        images.extend(message_images)

        for tool_call in message.get("tool_calls") or []:
            parsed = normalize_tool_call(tool_call)
            if parsed:
                tool_calls_by_id[parsed["id"]] = parsed

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            text = "\n".join(
                part
                for part in [
                    text,
                    "工具调用请求：",
                    json.dumps(tool_calls, ensure_ascii=False),
                ]
                if part
            )

        if role in {"tool", "function"}:
            tool_call_id = str(message.get("tool_call_id") or message.get("name") or "")
            tool_results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": text,
                    "name": str(message.get("name") or ""),
                }
            )
            tool_name = message.get("name") or message.get("tool_call_id") or "unknown"
            text = f"{tool_name} 返回：{text}"

        if text:
            message_blocks.append(f"{role.upper() or 'MESSAGE'}:\n{text}")

    tool_lines = []
    for result in tool_results:
        call = tool_calls_by_id.get(result["tool_call_id"], {})
        name = call.get("name") or result["name"] or result["tool_call_id"] or "tool"
        arguments = call.get("arguments") or {}
        tool_lines.append(
            f"- {name}({json.dumps(arguments, ensure_ascii=False)}) 返回：{result['content']}"
        )

    prompt_parts = [
        "你是一个基于本地工具执行结果回答用户的助手。",
        "请只给出面向用户的最终自然语言回答。",
        "不要展示 JSON、tool_calls、tool_call_id、内部消息格式或调试过程。",
    ]
    full_context = "\n\n".join(message_blocks).strip() or latest_user_text(messages)
    prompt_parts.extend(["完整消息上下文：", full_context])
    if tool_lines:
        prompt_parts.extend(["本地工具执行结果：", "\n".join(tool_lines)])
    else:
        prompt_parts.append("本地工具执行结果：无")

    return "\n\n".join(prompt_parts).strip(), images


async def parse_message_content(
    content: Any,
    tmp_dir: Path,
) -> tuple[list[str], list[Path]]:
    if isinstance(content, str):
        return [content], []

    if not isinstance(content, list):
        return [json.dumps(content, ensure_ascii=False)], []

    texts: list[str] = []
    images: list[Path] = []
    for part in content:
        if not isinstance(part, dict):
            texts.append(str(part))
            continue

        part_type = part.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            texts.append(str(part.get("text", "")))
            continue

        if part_type in {"image_url", "input_image"}:
            image_ref = part.get("image_url") or part.get("image")
            if isinstance(image_ref, dict):
                image_ref = image_ref.get("url")
            if not image_ref:
                raise openai_error(422, "image_url part is missing url.")
            images.append(await materialize_image(str(image_ref), tmp_dir))
            continue

        texts.append(json.dumps(part, ensure_ascii=False))

    return texts, images


async def materialize_image(image_ref: str, tmp_dir: Path) -> Path:
    if image_ref.startswith("data:"):
        return write_data_url_image(image_ref, tmp_dir)

    parsed = urllib.parse.urlparse(image_ref)
    if parsed.scheme in {"http", "https"}:
        return await download_image(image_ref, tmp_dir)

    if parsed.scheme == "file":
        return validate_existing_image(Path(urllib.request.url2pathname(parsed.path)))

    return validate_existing_image(Path(image_ref).expanduser())


def write_data_url_image(data_url: str, tmp_dir: Path) -> Path:
    header, sep, encoded = data_url.partition(",")
    if not sep or ";base64" not in header:
        raise openai_error(422, "Only base64 data URL images are supported.")

    media_type = header[5:].split(";", 1)[0] or "image/png"
    if not media_type.startswith("image/"):
        raise openai_error(422, "data URL must contain an image media type.")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise openai_error(422, "Invalid base64 image data.") from exc

    enforce_image_size_limit(len(payload))
    suffix = image_suffix(media_type)
    path = tmp_dir / f"image_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(payload)
    return path


async def download_image(url: str, tmp_dir: Path) -> Path:
    return await asyncio.to_thread(download_image_sync, url, tmp_dir)


def download_image_sync(url: str, tmp_dir: Path) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "chatgpt-bridge-api/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get_content_type()
            if content_type and not content_type.startswith("image/"):
                raise openai_error(422, f"URL does not point to an image: {content_type}")

            max_bytes = max_image_bytes()
            payload = response.read(max_bytes + 1)
            enforce_image_size_limit(len(payload))
    except urllib.error.URLError as exc:
        raise openai_error(422, f"Could not download image: {exc}") from exc

    suffix = image_suffix(content_type or "image/png")
    path = tmp_dir / f"image_{uuid.uuid4().hex}{suffix}"
    path.write_bytes(payload)
    return path


def validate_existing_image(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise openai_error(422, f"Image file does not exist: {resolved}")
    enforce_image_size_limit(resolved.stat().st_size)
    return resolved


def image_suffix(media_type: str) -> str:
    suffix = mimetypes.guess_extension(media_type.split(";", 1)[0]) or ".png"
    return ".jpg" if suffix == ".jpe" else suffix


def max_image_bytes() -> int:
    return env_int("BRIDGE_MAX_IMAGE_MB", 25) * 1024 * 1024


def enforce_image_size_limit(size: int) -> None:
    limit = max_image_bytes()
    if size > limit:
        raise openai_error(413, f"Image exceeds limit of {limit // 1024 // 1024} MB.")


def chat_completion_response(reply: str, request_model: str, prompt: str) -> dict[str, Any]:
    completion_tokens = estimate_tokens(reply)
    prompt_tokens = estimate_tokens(prompt)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def tool_call_response(
    tool_calls: list[dict[str, Any]],
    request_model: str,
    prompt: str,
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": estimate_tokens(prompt),
            "completion_tokens": 0,
            "total_tokens": estimate_tokens(prompt),
        },
    }


def responses_api_response(result: dict[str, Any], request_model: str) -> dict[str, Any]:
    prompt = str(result.get("prompt") or "")
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())

    if result["kind"] == "tool_calls":
        output = [responses_function_call_item(tool_call) for tool_call in result["tool_calls"]]
        output_text = ""
        output_tokens = 0
    else:
        output_text = str(result.get("reply") or "")
        output = [responses_message_item(output_text)]
        output_tokens = estimate_tokens(output_text)

    input_tokens = estimate_tokens(prompt)
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": request_model,
        "output": output,
        "output_text": output_text,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def responses_message_item(text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def responses_function_call_item(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call, dict) else {}
    function = function if isinstance(function, dict) else {}
    arguments = function.get("arguments") or "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": str(tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"),
        "name": str(function.get("name") or "unknown"),
        "arguments": arguments,
    }


async def stream_responses_response(response: dict[str, Any]) -> AsyncIterator[str]:
    response_id = str(response["id"])
    created = {**response, "status": "in_progress", "output": []}
    yield response_sse("response.created", {"type": "response.created", "response": created})

    for output_index, item in enumerate(response.get("output") or []):
        item_type = item.get("type")
        if item_type == "message":
            yield response_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item": {**item, "content": []},
                },
            )
            content = item.get("content") or []
            part = content[0] if content else {"type": "output_text", "text": "", "annotations": []}
            text = str(part.get("text") or "") if isinstance(part, dict) else ""
            yield response_sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "response_id": response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
            if text:
                yield response_sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                    },
                )
            yield response_sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                },
            )
            yield response_sse(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "response_id": response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part,
                },
            )
        elif item_type == "function_call":
            yield response_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item": {**item, "arguments": ""},
                },
            )
            arguments = str(item.get("arguments") or "")
            if arguments:
                yield response_sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": response_id,
                        "item_id": item["id"],
                        "output_index": output_index,
                        "delta": arguments,
                    },
                )
            yield response_sse(
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": arguments,
                },
            )

        yield response_sse(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": output_index,
                "item": item,
            },
        )

    yield response_sse("response.completed", {"type": "response.completed", "response": response})
    yield "data: [DONE]\n\n"


def response_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_response(reply: str, request_model: str) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [{"index": 0, "delta": {"content": reply}, "finish_reason": None}],
    }
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request_model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    for chunk in (first_chunk, content_chunk, final_chunk):
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llm_api_server:app",
        host=env_str("BRIDGE_HOST", "127.0.0.1") or "127.0.0.1",
        port=env_int("BRIDGE_PORT", 8000),
        reload=False,
    )


if __name__ == "__main__":
    main()
