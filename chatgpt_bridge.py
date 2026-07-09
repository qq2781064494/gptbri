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
import sys
import time
from pathlib import Path
from typing import Iterable

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

ASSISTANT_MESSAGE_SELECTORS = [
    '[data-message-author-role="assistant"]',
    '[data-testid*="conversation-turn"] [data-message-author-role="assistant"]',
    'article:has([data-message-author-role="assistant"])',
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


async def last_assistant_text(page: Page) -> str:
    """Extract text from the latest assistant message."""
    for selector in ASSISTANT_MESSAGE_SELECTORS:
        locator = page.locator(selector)
        try:
            count = await locator.count()
            if not count:
                continue

            message = locator.nth(count - 1)
            markdown = message.locator(".markdown").last
            if await markdown.count():
                text = await markdown.inner_text(timeout=1000)
            else:
                text = await message.inner_text(timeout=1000)

            text = cleanup_chatgpt_text(text)
            if text:
                return text
        except PlaywrightError:
            continue

    return ""


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


async def wait_for_response(page: Page, before_count: int, timeout_ms: int) -> str:
    """Wait for a new assistant answer and return its final stable text."""
    deadline = time.monotonic() + timeout_ms / 1000
    last_text = ""
    last_change_at = time.monotonic()

    while time.monotonic() < deadline:
        text = await last_assistant_text(page)
        count = await assistant_message_count(page)

        if count > before_count and text:
            if text != last_text:
                last_text = text
                last_change_at = time.monotonic()

            stop_button = await first_visible(page, STOP_BUTTON_SELECTORS)
            response_is_stable = time.monotonic() - last_change_at >= 2.0

            if response_is_stable and not stop_button:
                return last_text

        await page.wait_for_timeout(700)

    if last_text:
        return last_text

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

    async def _get_page(self, conversation_key: str = "default") -> tuple[Page, bool]:
        if not self._context:
            raise ChatGPTBridgeError("浏览器上下文尚未启动。")

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
    ) -> None:
        session = self._chat_sessions.setdefault(
            conversation_key,
            {"url": None, "started_at": None},
        )
        session_url = str(session["url"]) if session.get("url") else None
        expired = self._chat_expired(conversation_key)
        target_url = self.chat_url if force_new_chat or expired or not session_url else session_url

        if page.url != target_url:
            await page.goto(target_url, wait_until="domcontentloaded", timeout=self.timeout_ms)

        if force_new_chat or expired or not session.get("started_at"):
            session["started_at"] = time.monotonic()

    async def ask(
        self,
        text: str,
        images: list[Path] | None = None,
        conversation_key: str = "default",
        force_new_chat: bool = False,
    ) -> str:
        for attempt in range(2):
            try:
                return await self._ask_once(
                    text=text,
                    images=images,
                    conversation_key=conversation_key,
                    force_new_chat=force_new_chat,
                )
            except PlaywrightError as exc:
                if attempt == 0 and self._is_closed_browser_error(exc):
                    await self.reconnect()
                    continue
                raise ChatGPTBridgeError(f"浏览器自动化失败：{exc}") from exc

        raise ChatGPTBridgeError("浏览器自动化失败。")

    async def _ask_once(
        self,
        text: str,
        images: list[Path] | None = None,
        conversation_key: str = "default",
        force_new_chat: bool = False,
    ) -> str:
        if not self._context:
            raise ChatGPTBridgeError("浏览器上下文尚未启动。")

        page, should_close_page = await self._get_page(conversation_key=conversation_key)

        try:
            await self._ensure_active_chat(
                page,
                conversation_key=conversation_key,
                force_new_chat=force_new_chat,
            )
            await dismiss_blocking_modals(page)

            composer = await wait_for_composer(page, self.timeout_ms)
            before_count = await assistant_message_count(page)

            await attach_images(page, images or [], self.timeout_ms)
            await dismiss_blocking_modals(page)
            composer = await wait_for_composer(page, self.timeout_ms)
            await put_text_into_composer(page, composer, text)
            send_button = await wait_until_send_ready(page, self.timeout_ms)
            await dismiss_blocking_modals(page)
            try:
                await send_button.click()
            except PlaywrightError:
                await dismiss_blocking_modals(page)
                send_button = await wait_until_send_ready(page, self.timeout_ms)
                await send_button.click()

            reply = await wait_for_response(page, before_count, self.timeout_ms)
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
