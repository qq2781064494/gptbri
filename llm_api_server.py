#!/usr/bin/env python3
"""OpenAI-compatible local API server backed by the ChatGPT web bridge."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
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


def model_name() -> str:
    return env_str("BRIDGE_MODEL_NAME", DEFAULT_MODEL_NAME) or DEFAULT_MODEL_NAME


class Runtime:
    def __init__(self) -> None:
        self.bridge: ChatGPTWebBridge | None = None
        self.lock = asyncio.Lock()


runtime = Runtime()


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

    if runtime.bridge is None:
        raise openai_error(503, "Bridge is not ready.")

    tools = extract_tools(payload)
    tool_choice = payload.get("tool_choice", "auto")
    has_tools_result = has_tool_result(messages)
    conversation_key = request_conversation_key(tools, has_tools_result)
    if tools and tool_choice != "none" and not has_tools_result:
        with tempfile.TemporaryDirectory(prefix="chatgpt_bridge_api_") as tmp_dir:
            prompt, _ = await build_prompt_and_images(messages, Path(tmp_dir))
        tool_calls, final_answer = await choose_tool_response(
            messages,
            tools,
            tool_choice,
            conversation_key,
        )
        if tool_calls:
            return JSONResponse(tool_call_response(tool_calls, request_model, prompt))
        if final_answer:
            return JSONResponse(chat_completion_response(final_answer, request_model, prompt))

    with tempfile.TemporaryDirectory(prefix="chatgpt_bridge_api_") as tmp_dir:
        if has_tools_result:
            prompt, images = await build_final_tool_answer_prompt(messages, Path(tmp_dir))
        else:
            prompt, images = await build_prompt_and_images(messages, Path(tmp_dir))

        async with runtime.lock:
            try:
                reply = await runtime.bridge.ask(
                    text=prompt,
                    images=images,
                    conversation_key=conversation_key,
                )
            except ChatGPTBridgeError as exc:
                raise openai_error(502, str(exc)) from exc

    if payload.get("stream") is True:
        return StreamingResponse(
            stream_response(reply, request_model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return JSONResponse(chat_completion_response(reply, request_model, prompt))


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
) -> tuple[list[dict[str, Any]], str | None]:
    forced_name = forced_tool_name(tool_choice)
    planned = await plan_tool_calls_with_web(messages, tools, forced_name, conversation_key)
    if planned[0] or planned[1]:
        return planned

    if env_bool("BRIDGE_USE_LOCAL_TOOL_FALLBACK", False):
        deterministic = deterministic_tool_calls(messages, tools, forced_name)
        if deterministic:
            return deterministic, None

    return [], None


def forced_tool_name(tool_choice: Any) -> str | None:
    if not isinstance(tool_choice, dict):
        return None
    function = tool_choice.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return None


def deterministic_tool_calls(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    forced_name: str | None = None,
) -> list[dict[str, Any]]:
    query = latest_user_text(messages)
    if not query:
        return []

    available = {tool["name"].lower(): tool for tool in tools}
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
                if item.get("type") in {"text", "input_text"}:
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
) -> tuple[list[dict[str, Any]], str | None]:
    if runtime.bridge is None:
        return [], None

    user_text = latest_user_text(messages)
    planner_prompt = (
        "你是一个 OpenAI tool calling 规划器。你的任务是根据用户问题和工具定义，"
        "判断是否需要调用工具，并输出严格 JSON。\n\n"
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
        f"用户问题：\n{user_text}"
    )

    async with runtime.lock:
        try:
            reply = await runtime.bridge.ask(
                text=planner_prompt,
                images=[],
                conversation_key=conversation_key,
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
    user_blocks: list[str] = []
    system_blocks: list[str] = []
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

        if role == "system" and text:
            system_blocks.append(text)
        elif role == "user" and text:
            user_blocks.append(text)

        for tool_call in message.get("tool_calls") or []:
            parsed = normalize_tool_call(tool_call)
            if parsed:
                tool_calls_by_id[parsed["id"]] = parsed

        if role in {"tool", "function"}:
            tool_call_id = str(message.get("tool_call_id") or message.get("name") or "")
            tool_results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": text,
                    "name": str(message.get("name") or ""),
                }
            )

    user_text = "\n\n".join(user_blocks).strip()
    if not user_text:
        user_text = latest_user_text(messages)

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
    if system_blocks:
        prompt_parts.extend(["系统要求：", "\n\n".join(system_blocks)])
    prompt_parts.extend(["用户问题：", user_text])
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
        if part_type in {"text", "input_text"}:
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
