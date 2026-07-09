#!/usr/bin/env python3
"""Send text/images to a ChatGPT web conversation and return the reply.

This script automates chatgpt.com with Playwright. It intentionally uses a
dedicated persistent browser profile so you can log in once without storing
credentials in code.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    from playwright.async_api import (
        BrowserContext,
        Error as PlaywrightError,
        Locator,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ModuleNotFoundError as exc:
    BrowserContext = Locator = Page = object  # type: ignore[assignment]

    class PlaywrightError(Exception):
        pass

    class PlaywrightTimeoutError(PlaywrightError):
        pass

    async_playwright = None
    PLAYWRIGHT_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    PLAYWRIGHT_IMPORT_ERROR = None


DEFAULT_CHAT_URL = "https://chatgpt.com/c/6a4cc5f7-c524-83ee-a764-a38507e616fd"
DEFAULT_PROFILE_DIR = ".chatgpt_playwright_profile"
logger = logging.getLogger("chatgpt_bridge")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def detailed_logs_enabled() -> bool:
    return env_bool("BRIDGE_DETAILED_LOGS", False)


def detailed_log(message: str, *args: Any) -> None:
    if detailed_logs_enabled():
        logger.info(message, *args)


def response_force_return_stable_seconds() -> float:
    return env_float(
        "BRIDGE_RESPONSE_FORCE_RETURN_STABLE_SECONDS",
        RESPONSE_FORCE_RETURN_STABLE_SECONDS,
    )

COMPOSER_SELECTORS = [
    'div[contenteditable="true"][role="textbox"]',
    'div[contenteditable="true"]',
    '[data-testid="prompt-textarea"]',
    '#prompt-textarea',
    'textarea[placeholder*="Message"]',
    'textarea[placeholder*="消息"]',
    "textarea",
]

SEND_BUTTON_SELECTORS = [
    'button[data-testid="composer-submit-button"]',
    'button[data-testid="send-button"]',
    'button[aria-label*="Send"]',
    'button[aria-label*="Send prompt"]',
    'button[aria-label*="发送"]',
    'button[aria-label*="发送提示"]',
    'button[aria-label*="提交"]',
]

STOP_BUTTON_SELECTORS = [
    'button[data-testid="stop-button"]',
    'button[aria-label*="Stop"]',
    'button[aria-label*="停止"]',
]

ASSISTANT_DONE_SELECTORS = [
    'button[data-testid="copy-turn-action-button"]',
    'button[data-testid="good-response-turn-action-button"]',
    'button[data-testid="bad-response-turn-action-button"]',
    'button[aria-label*="Good response"]',
    'button[aria-label*="Bad response"]',
    'button[aria-label*="Read aloud"]',
    'button[aria-label*="朗读"]',
    'button[aria-label*="Regenerate"]',
    'button[aria-label*="重新生成"]',
    'button[aria-label*="More"]',
    'button[aria-label*="更多"]',
]

RESPONSE_STABLE_SECONDS = 2.0
RESPONSE_FORCE_RETURN_STABLE_SECONDS = 20.0
RESPONSE_TRACE_LOG_INTERVAL_SECONDS = 2.0
LOG_TEXT_PREVIEW_CHARS = 120
EXISTING_USER_MARK_ATTR = "data-chatgpt-bridge-existing-user"

ASSISTANT_MESSAGE_SELECTORS = [
    '[data-message-author-role="assistant"]',
    '[data-testid*="conversation-turn"] [data-message-author-role="assistant"]',
    'article:has([data-message-author-role="assistant"])',
]

FINAL_ASSISTANT_CONTENT_SELECTORS = [
    ".markdown",
    '[data-testid="markdown"]',
    ".prose",
]

DISMISS_MODAL_BUTTON_SELECTORS = [
    'button:has-text("明白了")',
    'button:has-text("知道了")',
    'button:has-text("确定")',
    'button:has-text("关闭")',
    'button:has-text("Got it")',
    'button:has-text("OK")',
    'button:has-text("Close")',
]


class ChatGPTBridgeError(RuntimeError):
    """Raised when the ChatGPT web automation cannot complete."""


async def first_visible(page: Page, selectors: Iterable[str]) -> Locator | None:
    """Return the first visible locator matching any selector."""
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = await locator.count()
            for index in range(count - 1, -1, -1):
                candidate = locator.nth(index)
                if await is_usable_visible(candidate):
                    return candidate
        except PlaywrightError:
            continue
    return None


async def is_usable_visible(locator: Locator) -> bool:
    """Return true only for elements that can actually receive user input."""
    try:
        box = await locator.bounding_box(timeout=700)
        if not box or box["width"] < 1 or box["height"] < 1:
            return False

        return await locator.evaluate(
            """el => {
                const style = getComputedStyle(el);
                const className = String(el.className || "");
                if (el.tagName === "TEXTAREA" && className.includes("fallbackTextarea")) {
                    return false;
                }
                return style.display !== "none"
                    && style.visibility !== "hidden"
                    && style.pointerEvents !== "none"
                    && !el.hidden
                    && el.getAttribute("aria-hidden") !== "true";
            }"""
        )
    except PlaywrightError:
        return False


async def any_locator(page: Page, selectors: Iterable[str]) -> Locator | None:
    """Return the first locator that exists, visible or hidden."""
    for selector in selectors:
        locator = page.locator(selector).last
        try:
            if await locator.count():
                return locator
        except PlaywrightError:
            continue
    return None


async def wait_for_composer(page: Page, timeout_ms: int) -> Locator:
    """Wait for the message composer, pausing for manual login if needed."""
    started_at = time.monotonic()
    deadline = time.monotonic() + timeout_ms / 1000
    login_prompted = False

    while time.monotonic() < deadline:
        composer = await first_visible(page, COMPOSER_SELECTORS)
        if composer:
            return composer

        if not login_prompted and time.monotonic() - started_at >= 8:
            print(
                "未找到 ChatGPT 输入框。如果页面要求登录，请在打开的浏览器里完成登录，"
                "然后回到终端按 Enter 继续。",
                file=sys.stderr,
            )
            login_prompted = True
            await asyncio.to_thread(input)

        await page.wait_for_timeout(1000)

    raise ChatGPTBridgeError("等待 ChatGPT 输入框超时，请确认已登录且页面可访问。")


async def dismiss_blocking_modals(page: Page) -> None:
    """Dismiss common ChatGPT modal overlays that block clicks."""
    for selector in DISMISS_MODAL_BUTTON_SELECTORS:
        locator = page.locator(selector)
        try:
            count = await locator.count()
            for index in range(count - 1, -1, -1):
                button = locator.nth(index)
                if await button.is_visible(timeout=300):
                    await button.click(timeout=1500)
                    await page.wait_for_timeout(500)
                    return
        except PlaywrightError:
            continue

    try:
        visible_dialog = page.locator('[role="dialog"], [data-testid*="modal"]').filter(
            has_text="请求过于频繁"
        )
        if await visible_dialog.count():
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(500)
    except PlaywrightError:
        pass


async def put_text_into_composer(page: Page, composer: Locator, text: str) -> None:
    """Fill textarea/contenteditable composer in a browser-compatible way."""
    tag_name = await composer.evaluate("el => el.tagName.toLowerCase()")

    await composer.scroll_into_view_if_needed()
    await composer.click(force=True)
    if tag_name == "textarea":
        await composer.fill(text)
        return

    await composer.evaluate(
        """(el, text) => {
            el.focus();
            const selection = window.getSelection();
            const range = document.createRange();
            range.selectNodeContents(el);
            selection.removeAllRanges();
            selection.addRange(range);

            document.execCommand("insertText", false, text);

            const currentText = (el.innerText || el.textContent || "").trim();
            if (!currentText) {
                el.innerHTML = "";
                const lines = text.split(/\\n/);
                for (const line of lines) {
                    const p = document.createElement("p");
                    p.textContent = line || " ";
                    el.appendChild(p);
                }
            }

            el.dispatchEvent(new InputEvent("input", {
                bubbles: true,
                cancelable: true,
                composed: true,
                inputType: "insertText",
                data: text,
            }));
            el.dispatchEvent(new Event("change", {bubbles: true}));
        }""",
        text,
    )

    await page.wait_for_timeout(300)
    inserted_text = await composer.evaluate("el => (el.innerText || el.textContent || '').trim()")
    if not inserted_text:
        await page.keyboard.insert_text(text)


async def attach_images(page: Page, image_paths: list[Path], timeout_ms: int) -> None:
    """Attach images to ChatGPT's composer."""
    if not image_paths:
        return

    file_payload = [str(path) for path in image_paths]

    file_input = page.locator('input[type="file"]').first
    if await file_input.count():
        await file_input.set_input_files(file_payload)
    else:
        attach_buttons = [
            'button[data-testid="composer-plus-btn"]',
            'button[data-testid="file-upload-button"]',
            'button[aria-label*="Attach"]',
            'button[aria-label*="Upload"]',
            'button[aria-label*="上传"]',
            'button[aria-label*="附加"]',
            'button[aria-label*="添加"]',
        ]
        button = await first_visible(page, attach_buttons)
        if not button:
            raise ChatGPTBridgeError("未找到图片上传控件。ChatGPT 页面结构可能已变化。")

        async with page.expect_file_chooser(timeout=timeout_ms) as chooser_info:
            await button.click()
        chooser = await chooser_info.value
        await chooser.set_files(file_payload)


async def wait_until_send_ready(page: Page, timeout_ms: int) -> Locator:
    """Wait until the send button exists and is enabled."""
    deadline = time.monotonic() + timeout_ms / 1000

    while time.monotonic() < deadline:
        send_button = await first_visible(page, SEND_BUTTON_SELECTORS)
        if send_button:
            try:
                if await send_button.is_enabled():
                    return send_button
            except PlaywrightError:
                pass
        await page.wait_for_timeout(500)

    raise ChatGPTBridgeError("发送按钮一直不可用，可能图片仍在上传或输入框未识别。")


async def assistant_message_count(page: Page) -> int:
    for selector in ASSISTANT_MESSAGE_SELECTORS:
        try:
            count = await page.locator(selector).count()
            if count:
                return count
        except PlaywrightError:
            continue
    return 0


async def last_assistant_message(page: Page) -> Locator | None:
    for selector in ASSISTANT_MESSAGE_SELECTORS:
        locator = page.locator(selector)
        try:
            count = await locator.count()
            if not count:
                continue

            return locator.nth(count - 1)
        except PlaywrightError:
            continue

    return None


async def last_assistant_text(page: Page) -> str:
    """Extract text from the latest assistant message."""
    message = await last_assistant_message(page)
    if not message:
        return ""

    try:
        for selector in FINAL_ASSISTANT_CONTENT_SELECTORS:
            content = message.locator(selector)
            try:
                count = await content.count()
            except PlaywrightError:
                continue
            for index in range(count - 1, -1, -1):
                try:
                    text = cleanup_chatgpt_text(await content.nth(index).inner_text(timeout=1000))
                except PlaywrightError:
                    continue
                if text:
                    return text
    except PlaywrightError:
        pass

    return ""


async def last_assistant_turn(page: Page) -> Locator | None:
    message = await last_assistant_message(page)
    if not message:
        return None

    ancestor_selectors = [
        'xpath=ancestor::*[contains(@data-testid, "conversation-turn")][1]',
        "xpath=ancestor::article[1]",
    ]
    for selector in ancestor_selectors:
        try:
            ancestor = message.locator(selector)
            if await ancestor.count():
                return ancestor.first
        except PlaywrightError:
            continue

    return message


async def last_assistant_done_actions_visible(page: Page) -> bool:
    turn = await last_assistant_turn(page)
    if not turn:
        return False

    for selector in ASSISTANT_DONE_SELECTORS:
        try:
            locator = turn.locator(selector)
            count = await locator.count()
            for index in range(count - 1, -1, -1):
                if await is_usable_visible(locator.nth(index)):
                    return True
        except PlaywrightError:
            continue

    return False


async def mark_existing_user_messages(page: Page) -> int:
    """Mark all user messages already on the page before sending a new prompt."""
    try:
        return int(await page.evaluate(
            """attr => {
                const nodes = Array.from(
                    document.querySelectorAll('[data-message-author-role="user"]')
                );
                nodes.forEach(el => el.setAttribute(attr, "1"));
                return nodes.length;
            }""",
            EXISTING_USER_MARK_ATTR,
        ))
    except PlaywrightError:
        return 0


async def current_turn_response_state(page: Page) -> dict[str, Any]:
    """Return the assistant state belonging to the latest unmarked user turn."""
    empty = {
        "text": "",
        "raw_text": "",
        "has_final_content": False,
        "last_user_text": "",
        "last_user_is_existing": True,
        "has_assistant_after_last_user": False,
        "assistant_done": False,
        "role_count": 0,
        "last_user_index": -1,
        "assistant_index": -1,
    }

    try:
        state = await page.evaluate(
            """({doneSelectors, existingUserAttr, finalContentSelectors}) => {
                const textOf = el => {
                    if (!el) return "";
                    return el.innerText || el.textContent || "";
                };
                const finalContentNodesOf = el => {
                    if (!el) return [];

                    for (const selector of finalContentSelectors) {
                        let nodes = [];
                        try {
                            nodes = Array.from(el.querySelectorAll(selector));
                        } catch {
                            nodes = [];
                        }
                        const uniqueNodes = [];
                        for (const node of nodes) {
                            if (!textOf(node).trim()) {
                                continue;
                            }
                            if (uniqueNodes.some(existing => existing === node || existing.contains(node))) {
                                continue;
                            }
                            for (let index = uniqueNodes.length - 1; index >= 0; index -= 1) {
                                if (node.contains(uniqueNodes[index])) {
                                    uniqueNodes.splice(index, 1);
                                }
                            }
                            uniqueNodes.push(node);
                        }
                        if (uniqueNodes.length) {
                            return uniqueNodes;
                        }
                    }

                    return [];
                };
                const finalTextOf = el => {
                    return finalContentNodesOf(el).map(textOf).join("\\n\\n");
                };
                const isVisible = el => {
                    if (!el || el.hidden || el.getAttribute("aria-hidden") === "true") {
                        return false;
                    }
                    const rect = el.getBoundingClientRect();
                    const style = getComputedStyle(el);
                    return rect.width > 0
                        && rect.height > 0
                        && style.display !== "none"
                        && style.visibility !== "hidden"
                        && style.pointerEvents !== "none";
                };
                const turnOf = el => {
                    let current = el;
                    for (let depth = 0; current && depth < 10; depth += 1) {
                        const testId = current.getAttribute("data-testid") || "";
                        if (testId.includes("conversation-turn")
                            || current.tagName.toLowerCase() === "article") {
                            return current;
                        }
                        current = current.parentElement;
                    }
                    return el;
                };

                const roleNodes = Array.from(
                    document.querySelectorAll("[data-message-author-role]")
                ).filter(el => {
                    const role = el.getAttribute("data-message-author-role");
                    return role === "user" || role === "assistant";
                });

                let lastUserIndex = -1;
                for (let index = roleNodes.length - 1; index >= 0; index -= 1) {
                    if (roleNodes[index].getAttribute("data-message-author-role") === "user") {
                        lastUserIndex = index;
                        break;
                    }
                }

                const lastUser = lastUserIndex >= 0 ? roleNodes[lastUserIndex] : null;
                let assistant = null;
                let assistantIndex = -1;
                for (let index = lastUserIndex + 1; index < roleNodes.length; index += 1) {
                    if (roleNodes[index].getAttribute("data-message-author-role") !== "assistant") {
                        continue;
                    }
                    assistant = roleNodes[index];
                    assistantIndex = index;
                    if (finalTextOf(assistant).trim() || textOf(assistant).trim()) {
                        break;
                    }
                }

                const finalText = finalTextOf(assistant);
                const rawText = textOf(assistant);
                const turn = assistant ? turnOf(assistant) : null;
                const assistantDone = !!turn && doneSelectors.some(selector => {
                    try {
                        return Array.from(turn.querySelectorAll(selector)).some(isVisible);
                    } catch {
                        return false;
                    }
                });

                return {
                    text: finalText,
                    raw_text: rawText,
                    has_final_content: !!finalText.trim(),
                    last_user_text: textOf(lastUser),
                    last_user_is_existing: !lastUser
                        || lastUser.getAttribute(existingUserAttr) === "1",
                    has_assistant_after_last_user: !!assistant
                        && lastUserIndex >= 0
                        && assistantIndex > lastUserIndex,
                    assistant_done: assistantDone,
                    role_count: roleNodes.length,
                    last_user_index: lastUserIndex,
                    assistant_index: assistantIndex,
                };
            }""",
            {
                "doneSelectors": ASSISTANT_DONE_SELECTORS,
                "existingUserAttr": EXISTING_USER_MARK_ATTR,
                "finalContentSelectors": FINAL_ASSISTANT_CONTENT_SELECTORS,
            },
        )
    except PlaywrightError:
        return empty

    if not isinstance(state, dict):
        return empty

    return {
        "text": cleanup_chatgpt_text(str(state.get("text") or "")),
        "raw_text": cleanup_chatgpt_text(str(state.get("raw_text") or "")),
        "has_final_content": bool(state.get("has_final_content")),
        "last_user_text": cleanup_chatgpt_text(str(state.get("last_user_text") or "")),
        "last_user_is_existing": bool(state.get("last_user_is_existing", True)),
        "has_assistant_after_last_user": bool(state.get("has_assistant_after_last_user")),
        "assistant_done": bool(state.get("assistant_done")),
        "role_count": int(state.get("role_count") or 0),
        "last_user_index": int(state.get("last_user_index") or -1),
        "assistant_index": int(state.get("assistant_index") or -1),
    }


async def chat_dom_snapshot(page: Page, limit: int = 8) -> dict[str, Any]:
    """Return a compact snapshot of the latest ChatGPT role nodes."""
    try:
        snapshot = await page.evaluate(
            """({existingUserAttr, finalContentSelectors, limit}) => {
                const textOf = el => {
                    if (!el) return "";
                    return (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
                };
                const finalContentNodesOf = el => {
                    if (!el) return [];

                    for (const selector of finalContentSelectors) {
                        let nodes = [];
                        try {
                            nodes = Array.from(el.querySelectorAll(selector));
                        } catch {
                            nodes = [];
                        }
                        const uniqueNodes = [];
                        for (const node of nodes) {
                            if (!textOf(node)) {
                                continue;
                            }
                            if (uniqueNodes.some(existing => existing === node || existing.contains(node))) {
                                continue;
                            }
                            for (let index = uniqueNodes.length - 1; index >= 0; index -= 1) {
                                if (node.contains(uniqueNodes[index])) {
                                    uniqueNodes.splice(index, 1);
                                }
                            }
                            uniqueNodes.push(node);
                        }
                        if (uniqueNodes.length) {
                            return uniqueNodes;
                        }
                    }

                    return [];
                };
                const finalTextOf = el => finalContentNodesOf(el).map(textOf).join("\\n\\n").trim();
                const tailOf = text => text.slice(Math.max(0, text.length - 80));
                const headOf = text => text.slice(0, 80);
                const hasSelector = (el, selector) => {
                    try {
                        return !!el && !!el.querySelector(selector);
                    } catch {
                        return false;
                    }
                };
                const turnOf = el => {
                    let current = el;
                    for (let depth = 0; current && depth < 10; depth += 1) {
                        const testId = current.getAttribute("data-testid") || "";
                        if (testId.includes("conversation-turn")
                            || current.tagName.toLowerCase() === "article") {
                            return current;
                        }
                        current = current.parentElement;
                    }
                    return el;
                };
                const roleNodes = Array.from(
                    document.querySelectorAll("[data-message-author-role]")
                ).filter(el => {
                    const role = el.getAttribute("data-message-author-role");
                    return role === "user" || role === "assistant";
                });
                const start = Math.max(0, roleNodes.length - limit);
                const nodes = roleNodes.slice(start).map((el, offset) => {
                    const rect = el.getBoundingClientRect();
                    const turn = turnOf(el);
                    const rawText = textOf(el);
                    const finalText = finalTextOf(el);
                    return {
                        index: start + offset,
                        role: el.getAttribute("data-message-author-role") || "",
                        existing_user: el.getAttribute(existingUserAttr) === "1",
                        has_final_content: !!finalText,
                        text_len: finalText.length,
                        text_head: headOf(finalText),
                        text_tail: tailOf(finalText),
                        raw_text_len: rawText.length,
                        raw_text_head: headOf(rawText),
                        raw_text_tail: tailOf(rawText),
                        final_selectors: finalContentSelectors.filter(selector => hasSelector(el, selector)),
                        visible: rect.width > 0 && rect.height > 0,
                        tag: el.tagName.toLowerCase(),
                        testid: el.getAttribute("data-testid") || "",
                        turn_testid: turn ? (turn.getAttribute("data-testid") || "") : "",
                    };
                });
                return {
                    url: location.href,
                    title: document.title,
                    role_count: roleNodes.length,
                    nodes,
                };
            }""",
            {
                "existingUserAttr": EXISTING_USER_MARK_ATTR,
                "finalContentSelectors": FINAL_ASSISTANT_CONTENT_SELECTORS,
                "limit": limit,
            },
        )
    except PlaywrightError as exc:
        return {"error": str(exc)}

    return snapshot if isinstance(snapshot, dict) else {"snapshot": snapshot}


def compact_for_match(text: str) -> str:
    return " ".join(text.split())


def preview_for_log(text: str, max_chars: int = LOG_TEXT_PREVIEW_CHARS) -> str:
    compacted = compact_for_match(text)
    if len(compacted) <= max_chars:
        return compacted

    half = max(1, (max_chars - 3) // 2)
    return f"{compacted[:half]}...{compacted[-half:]}"


def independent_request_marker(text: str) -> str:
    compacted = compact_for_match(text)
    marker_start = compacted.find("【独立请求 ")
    if marker_start < 0:
        return ""

    marker_end = compacted.find("】", marker_start)
    if marker_end < 0:
        return ""

    return compacted[marker_start : marker_end + 1]


def request_text_matches_user(sent_text: str, user_text: str) -> bool:
    sent = compact_for_match(sent_text)
    user = compact_for_match(user_text)
    if not sent or not user:
        return False

    marker = independent_request_marker(sent)
    if marker:
        return marker in user

    if sent == user or sent in user or user in sent:
        return True

    if len(sent) <= 120:
        return sent in user

    head = sent[:80]
    tail = sent[-80:]
    return head in user and tail in user


def is_transient_response_text(text: str) -> bool:
    """Return true for ChatGPT progress placeholders, not usable answers."""
    compacted = compact_for_match(text).strip()
    if not compacted:
        return False

    collapsed = re.sub(r"\s+", "", compacted).strip("。.!！…")
    lower_compacted = compacted.lower().strip(" .!…")
    lower_collapsed = collapsed.lower()

    chinese_patterns = [
        r"^正在(思考|生成|搜索|检索|读取|加载|上传|处理)$",
        r"^正在(分析|处理|读取|识别|查看|加载|上传)\d*(幅|张|个|份)?(图片|图像|文件|附件)$",
        r"^(分析|处理|读取|识别|查看|加载|上传)中$",
    ]
    if any(re.fullmatch(pattern, collapsed) for pattern in chinese_patterns):
        return True

    english_patterns = [
        r"^(thinking|working|generating|searching|reading|loading|uploading|processing)$",
        r"^(analyzing|processing|reading|loading|uploading)\d*(image|images|file|files|attachment|attachments)$",
        r"^(analyzing|processing|reading|loading|uploading)(an)?\d*(image|images|file|files|attachment|attachments)$",
    ]
    return any(re.fullmatch(pattern, lower_collapsed) for pattern in english_patterns) or (
        lower_compacted.startswith(("analyzing ", "processing ", "reading "))
        and lower_compacted.endswith((" image", " images", " file", " files", " attachment", " attachments"))
    )


def cleanup_chatgpt_text(text: str) -> str:
    """Remove common UI fragments that can appear in extracted ChatGPT text."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    ignored = {
        "复制",
        "Copy",
        "Good response",
        "Bad response",
        "Share",
        "编辑",
    }
    return "\n".join(line for line in lines if line.strip() not in ignored).strip()


async def wait_for_response(
    page: Page,
    before_count: int,
    before_text: str,
    sent_text: str,
    timeout_ms: int,
    conversation_key: str = "default",
    log_label: str = "-",
) -> str:
    """Wait for a new assistant answer and return its final stable text."""
    started_at = time.monotonic()
    deadline = time.monotonic() + timeout_ms / 1000
    last_text = ""
    last_change_at = time.monotonic()
    last_trace_log_at = 0.0
    last_dom_snapshot_log_at = 0.0
    last_trace_signature: tuple[Any, ...] | None = None
    response_started = False
    requires_user_match = bool(independent_request_marker(sent_text))
    detailed_log(
        (
            "bridge.wait.start id=%s key=%s timeout_ms=%s before_count=%s "
            "before_text_len=%s requires_marker=%s marker=%r sent_head=%r sent_tail=%r"
        ),
        log_label,
        conversation_key,
        timeout_ms,
        before_count,
        len(before_text),
        requires_user_match,
        independent_request_marker(sent_text),
        preview_for_log(sent_text[:200]),
        preview_for_log(sent_text[-200:]),
    )

    while time.monotonic() < deadline:
        current_state = await current_turn_response_state(page)
        final_text = str(current_state.get("text") or "")
        raw_text = str(current_state.get("raw_text") or "")
        user_text = str(current_state.get("last_user_text") or "")
        count = await assistant_message_count(page)
        user_matches = request_text_matches_user(sent_text, user_text)
        has_assistant_after_user = bool(current_state.get("has_assistant_after_last_user"))
        assistant_done = bool(current_state.get("assistant_done"))
        current_user_ready = (
            not current_state.get("last_user_is_existing", True)
            and (
                user_matches
                or (not requires_user_match and bool(user_text))
            )
        )
        has_final_content = bool(current_state.get("has_final_content")) and bool(final_text)
        raw_text_is_transient = is_transient_response_text(raw_text)
        raw_fallback_usable = (
            not has_final_content
            and current_user_ready
            and has_assistant_after_user
            and assistant_done
            and bool(raw_text)
            and not raw_text_is_transient
        )
        text = final_text if has_final_content else (raw_text if raw_fallback_usable else "")
        text_is_transient = is_transient_response_text(text)
        current_turn_has_assistant = current_user_ready and has_assistant_after_user
        current_turn_has_text = (
            current_turn_has_assistant
            and bool(text)
            and not text_is_transient
        )

        if not text and not current_user_ready:
            fallback_text = await last_assistant_text(page)
            if fallback_text and not is_transient_response_text(fallback_text):
                text = fallback_text
                text_is_transient = False

        has_new_response = (
            current_turn_has_assistant
            or count > before_count
            or (text and not text_is_transient and text != before_text)
        )
        response_started = response_started or bool(has_new_response)
        now = time.monotonic()
        elapsed_seconds = now - started_at
        if now - last_dom_snapshot_log_at >= RESPONSE_TRACE_LOG_INTERVAL_SECONDS:
            last_dom_snapshot_log_at = now
            snapshot = await chat_dom_snapshot(page)
            detailed_log(
                "bridge.dom.snapshot id=%s key=%s elapsed_ms=%.1f snapshot=%s",
                log_label,
                conversation_key,
                elapsed_seconds * 1000,
                json.dumps(snapshot, ensure_ascii=False),
            )
        if response_started and text and not text_is_transient:
            if text != last_text:
                last_text = text
                last_change_at = now

            if not current_turn_has_text:
                assistant_done = await last_assistant_done_actions_visible(page)
            stop_button = await first_visible(page, STOP_BUTTON_SELECTORS)
            send_button = await first_visible(page, SEND_BUTTON_SELECTORS)
            stable_seconds = now - last_change_at
            response_is_stable = stable_seconds >= RESPONSE_STABLE_SECONDS

            response_looks_done = (
                assistant_done
                or not stop_button
                or send_button
                or stable_seconds >= response_force_return_stable_seconds()
            )
            trace_signature = (
                response_started,
                current_turn_has_text,
                current_user_ready,
                current_state.get("last_user_is_existing"),
                user_matches,
                bool(current_state.get("has_assistant_after_last_user")),
                has_final_content,
                len(text),
                len(raw_text),
                text_is_transient,
                raw_text_is_transient,
                raw_fallback_usable,
                count,
                assistant_done,
                bool(stop_button),
                bool(send_button),
                response_looks_done,
            )
            if (
                trace_signature != last_trace_signature
                or now - last_trace_log_at >= RESPONSE_TRACE_LOG_INTERVAL_SECONDS
            ):
                last_trace_signature = trace_signature
                last_trace_log_at = now
                detailed_log(
                    (
                        "bridge.wait.trace id=%s key=%s elapsed_ms=%.1f count=%s "
                        "before_count=%s role_count=%s last_user_index=%s assistant_index=%s "
                        "started=%s current_turn=%s user_ready=%s user_existing=%s "
                        "user_matches=%s requires_marker=%s assistant_after_user=%s "
                        "has_final=%s text_len=%s raw_text_len=%s "
                        "transient_text=%s transient_raw=%s raw_fallback=%s "
                        "stable_seconds=%.1f assistant_done=%s "
                        "stop_visible=%s send_visible=%s looks_done=%s "
                        "last_user_head=%r assistant_final_head=%r assistant_final_tail=%r "
                        "assistant_raw_head=%r"
                    ),
                    log_label,
                    conversation_key,
                    elapsed_seconds * 1000,
                    count,
                    before_count,
                    current_state.get("role_count"),
                    current_state.get("last_user_index"),
                    current_state.get("assistant_index"),
                    response_started,
                    current_turn_has_text,
                    current_user_ready,
                    current_state.get("last_user_is_existing"),
                    user_matches,
                    requires_user_match,
                    current_state.get("has_assistant_after_last_user"),
                    has_final_content,
                    len(text),
                    len(raw_text),
                    text_is_transient,
                    raw_text_is_transient,
                    raw_fallback_usable,
                    stable_seconds,
                    assistant_done,
                    bool(stop_button),
                    bool(send_button),
                    response_looks_done,
                    preview_for_log(user_text),
                    preview_for_log(text[:200]),
                    preview_for_log(text[-200:]),
                    preview_for_log(raw_text[:200]),
                )
            if response_is_stable and response_looks_done:
                detailed_log(
                    (
                        "bridge.wait.return id=%s key=%s elapsed_ms=%.1f reply_len=%s "
                        "stable_seconds=%.1f assistant_done=%s stop_visible=%s send_visible=%s"
                    ),
                    log_label,
                    conversation_key,
                    elapsed_seconds * 1000,
                    len(last_text),
                    stable_seconds,
                    assistant_done,
                    bool(stop_button),
                    bool(send_button),
                )
                return last_text
        elif now - last_trace_log_at >= RESPONSE_TRACE_LOG_INTERVAL_SECONDS:
            last_trace_log_at = now
            detailed_log(
                (
                    "bridge.wait.trace id=%s key=%s elapsed_ms=%.1f count=%s "
                    "before_count=%s role_count=%s last_user_index=%s assistant_index=%s "
                    "started=%s has_new_response=%s current_turn=%s user_ready=%s "
                    "user_existing=%s user_matches=%s requires_marker=%s assistant_after_user=%s "
                    "has_final=%s text_len=%s raw_text_len=%s transient_text=%s "
                    "transient_raw=%s raw_fallback=%s last_user_head=%r "
                    "assistant_final_head=%r assistant_raw_head=%r"
                ),
                log_label,
                conversation_key,
                elapsed_seconds * 1000,
                count,
                before_count,
                current_state.get("role_count"),
                current_state.get("last_user_index"),
                current_state.get("assistant_index"),
                response_started,
                bool(has_new_response),
                current_turn_has_text,
                current_user_ready,
                current_state.get("last_user_is_existing"),
                user_matches,
                requires_user_match,
                current_state.get("has_assistant_after_last_user"),
                has_final_content,
                len(text),
                len(raw_text),
                text_is_transient,
                raw_text_is_transient,
                raw_fallback_usable,
                preview_for_log(user_text),
                preview_for_log(text[:200]),
                preview_for_log(raw_text[:200]),
            )

        await page.wait_for_timeout(700)

    if last_text:
        detailed_log(
            "bridge.wait.timeout_return_last id=%s key=%s reply_len=%s timeout_ms=%s",
            log_label,
            conversation_key,
            len(last_text),
            timeout_ms,
        )
        return last_text

    logger.error(
        "bridge.wait.timeout_error id=%s key=%s timeout_ms=%s before_count=%s before_text_len=%s",
        log_label,
        conversation_key,
        timeout_ms,
        before_count,
        len(before_text),
    )
    raise ChatGPTBridgeError("等待 ChatGPT 输出超时。")


class ChatGPTWebBridge:
    def __init__(
        self,
        chat_url: str = DEFAULT_CHAT_URL,
        profile_dir: str | Path = DEFAULT_PROFILE_DIR,
        browser_channel: str | None = None,
        cdp_url: str | None = None,
        reuse_cdp_page: bool = True,
        close_extra_chatgpt_pages: bool = False,
        chat_reset_seconds: int = 0,
        retry_attempts: int = 1,
        headless: bool = False,
        timeout_ms: int = 180_000,
    ) -> None:
        self.chat_url = chat_url
        self.profile_dir = Path(profile_dir).expanduser().resolve()
        self.browser_channel = browser_channel
        self.cdp_url = cdp_url
        self.reuse_cdp_page = reuse_cdp_page
        self.close_extra_chatgpt_pages = close_extra_chatgpt_pages
        self.chat_reset_seconds = max(0, chat_reset_seconds)
        self.retry_attempts = max(0, retry_attempts)
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._context: BrowserContext | None = None
        self._browser = None
        self._playwright = None
        self._page: Page | None = None
        self._pages: dict[str, Page] = {}
        self._chat_sessions: dict[str, dict[str, float | str | None]] = {}
        self._closed_extra_pages = False

    async def __aenter__(self) -> "ChatGPTWebBridge":
        if async_playwright is None:
            raise ChatGPTBridgeError(
                "缺少 Playwright 依赖。请先运行："
                "pip install -r requirements.txt && python -m playwright install chromium"
            )

        self._playwright = await async_playwright().start()
        await self._open_context()
        return self

    async def _open_context(self) -> None:
        if not self._playwright:
            raise ChatGPTBridgeError("Playwright 尚未启动。")

        if self.cdp_url:
            self._browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
            if not self._browser.contexts:
                raise ChatGPTBridgeError("已连接 Chrome，但没有可用浏览器上下文。")
            self._context = self._browser.contexts[0]
            return

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        launch_options = {
            "user_data_dir": str(self.profile_dir),
            "headless": self.headless,
            "viewport": {"width": 1400, "height": 980},
            "locale": "zh-CN",
            "accept_downloads": True,
        }
        if self.browser_channel:
            launch_options["channel"] = self.browser_channel

        self._context = await self._playwright.chromium.launch_persistent_context(
            **launch_options
        )

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._close_context()
        if self._playwright:
            await self._playwright.stop()
        self._playwright = None

    async def _close_context(self) -> None:
        if self._context and not self.cdp_url:
            with contextlib.suppress(PlaywrightError):
                await self._context.close()

        self._context = None
        self._browser = None
        self._page = None
        self._pages.clear()
        self._closed_extra_pages = False

    async def reconnect(self) -> None:
        """Reconnect to Chrome after the CDP/browser context has gone away."""
        await self._close_context()
        try:
            await self._open_context()
        except PlaywrightError as exc:
            raise ChatGPTBridgeError(f"重新连接 Chrome 远程调试端口失败：{exc}") from exc

    @staticmethod
    def _is_closed_browser_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "target page, context or browser has been closed" in message
            or "browsercontext.new_page" in message and "closed" in message
            or "page.goto" in message and "closed" in message
            or "browser has been closed" in message
        )

    async def ensure_conversation_pages(self, conversation_keys: list[str]) -> None:
        for attempt in range(2):
            try:
                for conversation_key in conversation_keys:
                    page, _ = await self._get_page(conversation_key=conversation_key)
                    await self._ensure_active_chat(page, conversation_key=conversation_key)
                return
            except PlaywrightError as exc:
                if attempt == 0 and self._is_closed_browser_error(exc):
                    await self.reconnect()
                    continue
                raise ChatGPTBridgeError(f"浏览器自动化失败：{exc}") from exc

    async def _get_page(
        self,
        conversation_key: str = "default",
        force_new_page: bool = False,
    ) -> tuple[Page, bool]:
        if not self._context:
            raise ChatGPTBridgeError("浏览器上下文尚未启动。")

        if force_new_page:
            current = self._pages.pop(conversation_key, None)
            if conversation_key == "default":
                self._page = None
            self._chat_sessions.pop(conversation_key, None)
            if current and not current.is_closed():
                with contextlib.suppress(PlaywrightError):
                    await current.close()

            page = await self._context.new_page()
            self._pages[conversation_key] = page
            if conversation_key == "default":
                self._page = page
            return page, False

        if self.cdp_url and self.reuse_cdp_page:
            current = self._pages.get(conversation_key)
            if current and not current.is_closed():
                return current, False

            assigned_pages = {
                page
                for page in self._pages.values()
                if page and not page.is_closed()
            }
            chatgpt_pages = [
                page
                for page in self._context.pages
                if "chatgpt.com" in page.url and not page.is_closed() and page not in assigned_pages
            ]
            page = chatgpt_pages[0] if chatgpt_pages else await self._context.new_page()
            self._pages[conversation_key] = page
            if conversation_key == "default":
                self._page = page

            if self.close_extra_chatgpt_pages and not self._closed_extra_pages:
                self._closed_extra_pages = True
                for extra_page in chatgpt_pages[1:]:
                    try:
                        await extra_page.close()
                    except PlaywrightError:
                        pass

            return page, False

        if self.cdp_url:
            return await self._context.new_page(), True

        if self.reuse_cdp_page:
            current = self._pages.get(conversation_key)
            if current and not current.is_closed():
                return current, False
            page = self._context.pages[0] if not self._pages and self._context.pages else await self._context.new_page()
            self._pages[conversation_key] = page
            return page, False

        page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        return page, False

    async def close_conversation_page(
        self,
        conversation_key: str = "default",
        log_label: str = "-",
    ) -> None:
        current = self._pages.pop(conversation_key, None)
        if conversation_key == "default":
            self._page = None
        self._chat_sessions.pop(conversation_key, None)

        closed = False
        if current and not current.is_closed():
            with contextlib.suppress(PlaywrightError):
                await current.close()
                closed = True

        detailed_log(
            "bridge.page.closed id=%s key=%s closed=%s",
            log_label,
            conversation_key,
            closed,
        )

    async def _replace_page(
        self,
        conversation_key: str = "default",
        log_label: str = "-",
    ) -> None:
        if not self._context:
            raise ChatGPTBridgeError("浏览器上下文尚未启动。")

        current = self._pages.pop(conversation_key, None)
        if conversation_key == "default":
            self._page = None
        self._chat_sessions.pop(conversation_key, None)

        if current and not current.is_closed():
            with contextlib.suppress(PlaywrightError):
                await current.close()

        page = await self._context.new_page()
        self._pages[conversation_key] = page
        if conversation_key == "default":
            self._page = page
        detailed_log(
            "bridge.page.replaced id=%s key=%s new_url=%r",
            log_label,
            conversation_key,
            page.url,
        )

    def _chat_expired(self, conversation_key: str) -> bool:
        session = self._chat_sessions.get(conversation_key)
        if not session or session.get("started_at") is None:
            return True
        if self.chat_reset_seconds <= 0:
            return False
        return time.monotonic() - float(session["started_at"]) >= self.chat_reset_seconds

    async def _ensure_active_chat(
        self,
        page: Page,
        conversation_key: str = "default",
        force_new_chat: bool = False,
        log_label: str = "-",
    ) -> None:
        session = self._chat_sessions.setdefault(
            conversation_key,
            {"url": None, "started_at": None},
        )
        session_url = str(session["url"]) if session.get("url") else None
        expired = self._chat_expired(conversation_key)
        target_url = self.chat_url if force_new_chat or expired or not session_url else session_url
        detailed_log(
            (
                "bridge.chat.ensure id=%s key=%s force_new=%s expired=%s "
                "current_url=%r target_url=%r session_url=%r started=%s"
            ),
            log_label,
            conversation_key,
            force_new_chat,
            expired,
            page.url,
            target_url,
            session_url,
            bool(session.get("started_at")),
        )

        if page.url != target_url:
            detailed_log(
                "bridge.chat.goto.start id=%s key=%s from_url=%r to_url=%r",
                log_label,
                conversation_key,
                page.url,
                target_url,
            )
            await page.goto(target_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            detailed_log(
                "bridge.chat.goto.done id=%s key=%s url=%r",
                log_label,
                conversation_key,
                page.url,
            )

        if force_new_chat or expired or not session.get("started_at"):
            session["started_at"] = time.monotonic()

    async def ask(
        self,
        text: str,
        images: list[Path] | None = None,
        conversation_key: str = "default",
        force_new_chat: bool = False,
        force_new_page: bool = False,
        log_label: str = "-",
    ) -> str:
        last_error: BaseException | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                reply = await self._ask_once(
                    text=text,
                    images=images,
                    conversation_key=conversation_key,
                    force_new_chat=force_new_chat or attempt > 0,
                    force_new_page=force_new_page and attempt == 0,
                    log_label=log_label,
                )
                if not reply.strip():
                    raise ChatGPTBridgeError("ChatGPT 返回为空。")
                return reply
            except ChatGPTBridgeError as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    raise
                detailed_log(
                    "bridge.ask.retry id=%s key=%s attempt=%s retry_attempts=%s error=%r",
                    log_label,
                    conversation_key,
                    attempt + 1,
                    self.retry_attempts,
                    str(exc),
                )
                await self._replace_page(conversation_key=conversation_key, log_label=log_label)
            except PlaywrightError as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    raise ChatGPTBridgeError(f"浏览器自动化失败：{exc}") from exc
                detailed_log(
                    "bridge.ask.retry id=%s key=%s attempt=%s retry_attempts=%s error=%r",
                    log_label,
                    conversation_key,
                    attempt + 1,
                    self.retry_attempts,
                    str(exc),
                )
                if self._is_closed_browser_error(exc):
                    await self.reconnect()
                else:
                    await self._replace_page(conversation_key=conversation_key, log_label=log_label)

        if last_error:
            raise ChatGPTBridgeError(f"浏览器自动化失败：{last_error}") from last_error
        raise ChatGPTBridgeError("浏览器自动化失败。")

    async def _ask_once(
        self,
        text: str,
        images: list[Path] | None = None,
        conversation_key: str = "default",
        force_new_chat: bool = False,
        force_new_page: bool = False,
        log_label: str = "-",
    ) -> str:
        if not self._context:
            raise ChatGPTBridgeError("浏览器上下文尚未启动。")

        detailed_log(
            (
                "bridge.ask.start id=%s key=%s force_new=%s images=%s text_chars=%s "
                "marker=%r text_head=%r text_tail=%r"
            ),
            log_label,
            conversation_key,
            force_new_chat,
            len(images or []),
            len(text),
            independent_request_marker(text),
            preview_for_log(text[:200]),
            preview_for_log(text[-200:]),
        )
        page, should_close_page = await self._get_page(
            conversation_key=conversation_key,
            force_new_page=force_new_page,
        )
        detailed_log(
            "bridge.page.selected id=%s key=%s should_close=%s url=%r",
            log_label,
            conversation_key,
            should_close_page,
            page.url,
        )

        try:
            await self._ensure_active_chat(
                page,
                conversation_key=conversation_key,
                force_new_chat=force_new_chat,
                log_label=log_label,
            )
            await dismiss_blocking_modals(page)

            composer = await wait_for_composer(page, self.timeout_ms)
            before_count = await assistant_message_count(page)
            before_text = await last_assistant_text(page)
            marked_users = await mark_existing_user_messages(page)
            detailed_log(
                (
                    "bridge.dom.before_send id=%s key=%s before_count=%s marked_users=%s "
                    "before_text_len=%s before_text_head=%r url=%r"
                ),
                log_label,
                conversation_key,
                before_count,
                marked_users,
                len(before_text),
                preview_for_log(before_text[:200]),
                page.url,
            )

            await attach_images(page, images or [], self.timeout_ms)
            await dismiss_blocking_modals(page)
            composer = await wait_for_composer(page, self.timeout_ms)
            await put_text_into_composer(page, composer, text)
            detailed_log(
                "bridge.composer.filled id=%s key=%s text_chars=%s images=%s",
                log_label,
                conversation_key,
                len(text),
                len(images or []),
            )
            send_button = await wait_until_send_ready(page, self.timeout_ms)
            await dismiss_blocking_modals(page)
            try:
                await send_button.click()
            except PlaywrightError:
                await dismiss_blocking_modals(page)
                send_button = await wait_until_send_ready(page, self.timeout_ms)
                await send_button.click()
            after_send_state = await current_turn_response_state(page)
            after_send_user_text = str(after_send_state.get("last_user_text") or "")
            detailed_log(
                (
                    "bridge.send.clicked id=%s key=%s url=%r role_count=%s "
                    "last_user_index=%s assistant_index=%s user_existing=%s "
                    "user_matches=%s assistant_after_user=%s has_final=%s "
                    "assistant_final_len=%s assistant_raw_len=%s "
                    "last_user_head=%r assistant_final_head=%r assistant_raw_head=%r"
                ),
                log_label,
                conversation_key,
                page.url,
                after_send_state.get("role_count"),
                after_send_state.get("last_user_index"),
                after_send_state.get("assistant_index"),
                after_send_state.get("last_user_is_existing"),
                request_text_matches_user(text, after_send_user_text),
                after_send_state.get("has_assistant_after_last_user"),
                after_send_state.get("has_final_content"),
                len(str(after_send_state.get("text") or "")),
                len(str(after_send_state.get("raw_text") or "")),
                preview_for_log(after_send_user_text),
                preview_for_log(str(after_send_state.get("text") or "")),
                preview_for_log(str(after_send_state.get("raw_text") or "")),
            )
            detailed_log(
                "bridge.dom.snapshot id=%s key=%s elapsed_ms=0.0 snapshot=%s",
                log_label,
                conversation_key,
                json.dumps(await chat_dom_snapshot(page), ensure_ascii=False),
            )

            reply = await wait_for_response(
                page,
                before_count,
                before_text,
                text,
                self.timeout_ms,
                conversation_key=conversation_key,
                log_label=log_label,
            )
            detailed_log(
                "bridge.ask.return id=%s key=%s reply_chars=%s url=%r",
                log_label,
                conversation_key,
                len(reply),
                page.url,
            )
            self._chat_sessions.setdefault(
                conversation_key,
                {"url": None, "started_at": time.monotonic()},
            )
            self._chat_sessions[conversation_key]["url"] = page.url
            if not self._chat_sessions[conversation_key].get("started_at"):
                self._chat_sessions[conversation_key]["started_at"] = time.monotonic()
            return reply
        finally:
            if should_close_page:
                try:
                    await page.close()
                except PlaywrightError:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把文本和图片发送到指定 ChatGPT 网页会话，并把最新回复返回到程序。"
    )
    parser.add_argument("--chat-url", default=DEFAULT_CHAT_URL, help="目标 ChatGPT 会话 URL。")
    parser.add_argument("--text", default="", help="要发送的文本。")
    parser.add_argument("--text-file", help="从文件读取要发送的文本。")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="要上传的图片路径；可重复传多个 --image。",
    )
    parser.add_argument(
        "--profile-dir",
        default=DEFAULT_PROFILE_DIR,
        help="保存 ChatGPT 登录态的浏览器资料夹。",
    )
    parser.add_argument(
        "--browser-channel",
        choices=["chrome", "msedge"],
        help="使用系统浏览器通道。Google 登录提示浏览器不安全时推荐 chrome。",
    )
    parser.add_argument(
        "--cdp-url",
        help=(
            "连接已手动启动并开启 remote debugging 的 Chrome，例如 "
            "http://127.0.0.1:9222。安全认证卡住时推荐这个模式。"
        ),
    )
    parser.add_argument(
        "--new-page-per-request",
        action="store_true",
        help="CDP 模式下每次请求新开标签页。默认复用同一个 ChatGPT 标签页。",
    )
    parser.add_argument(
        "--close-extra-chatgpt-pages",
        action="store_true",
        help="CDP 模式下启动后关闭多余 ChatGPT 标签页，只保留一个页面。",
    )
    parser.add_argument(
        "--chat-reset-seconds",
        type=int,
        default=0,
        help="同一个 ChatGPT 对话复用多久后自动重开；0 表示不按时间自动重开。",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=1,
        help="回复为空或网页自动化失败后，打开新标签页重试的次数；0 表示不重试。",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式。第一次登录或遇到验证时不要开启。",
    )
    parser.add_argument("--timeout", type=int, default=180, help="总等待秒数。")
    parser.add_argument(
        "--conversation-key",
        default="default",
        help="复用的 ChatGPT 对话槽名称，例如 chat 或 tool。",
    )
    parser.add_argument(
        "--force-new-chat",
        action="store_true",
        help="本次调用强制开启新的 ChatGPT 对话。",
    )
    parser.add_argument("--output", help="把回复写入文本文件。")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果。")
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    chunks = []
    if args.text_file:
        chunks.append(Path(args.text_file).expanduser().read_text(encoding="utf-8"))
    if args.text:
        chunks.append(args.text)

    text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
    if not text:
        raise ChatGPTBridgeError("请通过 --text 或 --text-file 提供输入文本。")
    return text


def validate_images(paths: list[str]) -> list[Path]:
    images = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise ChatGPTBridgeError(f"图片不存在：{path}")
        if not path.is_file():
            raise ChatGPTBridgeError(f"图片路径不是文件：{path}")
        images.append(path)
    return images


async def async_main() -> int:
    args = parse_args()
    text = load_text(args)
    images = validate_images(args.image)

    async with ChatGPTWebBridge(
        chat_url=args.chat_url,
        profile_dir=args.profile_dir,
        browser_channel=args.browser_channel,
        cdp_url=args.cdp_url,
        reuse_cdp_page=not args.new_page_per_request,
        close_extra_chatgpt_pages=args.close_extra_chatgpt_pages,
        chat_reset_seconds=args.chat_reset_seconds,
        retry_attempts=args.retry_attempts,
        headless=args.headless,
        timeout_ms=args.timeout * 1000,
    ) as bridge:
        reply = await bridge.ask(
            text=text,
            images=images,
            conversation_key=args.conversation_key,
            force_new_chat=args.force_new_chat,
        )

    if args.output:
        Path(args.output).expanduser().write_text(reply, encoding="utf-8")

    if args.json:
        print(json.dumps({"reply": reply}, ensure_ascii=False, indent=2))
    else:
        print(reply)

    return 0


def main() -> int:
    try:
        return asyncio.run(async_main())
    except (ChatGPTBridgeError, PlaywrightTimeoutError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
