# ChatGPT Web Bridge API

把已登录的 ChatGPT 网页，桥接成本地 OpenAI 兼容 API。

你现有的 LangChain、OpenAI SDK、curl 或业务代码，不用改调用方式，继续按熟悉的三件套：

```text
api_key + base_url + model
```

服务会在本机启动一个兼容接口：

```text
http://127.0.0.1:8011/v1/chat/completions
```

然后把请求转发到一个已登录的 ChatGPT 浏览器窗口，再把网页回复包装成 OpenAI 风格响应返回。

> 说明：这不是 OpenAI 官方 API，而是本机网页自动化桥接。它适合个人本机开发、调试和把网页 ChatGPT 临时接入现有大模型代码，不建议暴露到公网或作为高并发生产服务。

## 它解决什么

很多大模型代码已经写成了这种形式：

```python
model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

这个项目的目标就是让 ChatGPT 网页也能被当成一个“大模型 API”来用：

```text
业务代码 / Agent / LangChain / OpenAI SDK
        ↓
本地 OpenAI 兼容接口
        ↓
FastAPI Bridge
        ↓
Playwright 连接已登录 Chrome
        ↓
ChatGPT 网页
```

## 功能特性

- OpenAI 兼容：支持 `/v1/chat/completions`、`/v1/models`、`Authorization: Bearer ...`
- LangChain 兼容：支持 `ChatOpenAI(...)`、`invoke(...)`、`bind_tools(...)`
- Tool Calling：让模型返回工具调用 JSON，本地执行 Python 工具，再把工具结果交回模型生成最终回答
- 图片输入：支持本地图片路径、`file://`、HTTP 图片 URL、base64 data URL
- 会话管理：普通聊天和工具调用分别使用固定 ChatGPT 标签页，避免每次请求新开页面
- 自动重开：可配置一个聊天框使用多久后重开，减少上下文污染
- 串行队列：同一时间只操作一个网页请求，避免输入框串线
- 本机优先：API key 只用于本地服务鉴权，不需要把 ChatGPT 登录信息写进代码

## 快速开始

### 1. 安装依赖

推荐使用项目已有的 `gptbri` conda 环境：

```bash
cd /Users/weijiaxin/Documents/pythonwork/gptbri
conda activate gptbri
pip install -r requirements.txt
python -m playwright install chromium
```

如果还没有环境：

```bash
conda create -n gptbri python=3.11
conda activate gptbri
cd /Users/weijiaxin/Documents/pythonwork/gptbri
pip install -r requirements.txt
python -m playwright install chromium
```

### 2. 配置 `.env`

复制或创建 `.env`：

```env
OPENAI_MODEL=chatgpt-web
OPENAI_BASE_URL=http://127.0.0.1:8011/v1
OPENAI_API_KEY=sk-local-test

BRIDGE_API_KEY=sk-local-test
BRIDGE_MODEL_NAME=chatgpt-web
BRIDGE_CDP_URL=http://127.0.0.1:9222
BRIDGE_PORT=8011
BRIDGE_CHAT_URL=https://chatgpt.com/
BRIDGE_REUSE_CDP_PAGE=1
BRIDGE_CLOSE_EXTRA_CHATGPT_PAGES=1
BRIDGE_CHAT_RESET_SECONDS=1800
BRIDGE_PREOPEN_CONVERSATION_PAGES=1
BRIDGE_CHAT_CONVERSATION_KEY=chat
BRIDGE_TOOL_CONVERSATION_KEY=tool
```

两组变量的分工：

```text
OPENAI_*   给你的业务代码、LangChain、OpenAI SDK 使用
BRIDGE_*   给本地桥接服务和浏览器自动化使用
```

### 3. 启动 Chrome 和 API 服务

第一个终端，启动带调试端口的专用 Chrome：

```bash
cd /Users/weijiaxin/Documents/pythonwork/gptbri
bash launch_chrome_debug.sh
```

在打开的 Chrome 里手动登录 ChatGPT，直到能正常看到输入框。这个 Chrome 窗口不要关闭。

第二个终端，启动本地 API：

```bash
cd /Users/weijiaxin/Documents/pythonwork/gptbri
conda activate gptbri
bash run_api_server.sh
```

检查服务：

```bash
curl http://127.0.0.1:8011/health
```

正常会返回类似：

```json
{
  "status": "ok",
  "model": "chatgpt-web",
  "cdp_url": "http://127.0.0.1:9222",
  "chat_reset_seconds": 1800,
  "chat_conversation_key": "chat",
  "tool_conversation_key": "tool"
}
```

## 调用示例

### LangChain

业务侧只需要像调用普通 OpenAI 兼容模型一样调用：

```python
from env_setup import load_project_env
load_project_env()

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
    temperature=0.7,
)

prompt_template = PromptTemplate(
    input_variables=["product"],
    template="为{product}写三个吸引人的广告语，需要面向年青人",
)

prompt = prompt_template.invoke({"product": "HideOnBoss"})
response = model.invoke(prompt)
answer = StrOutputParser().invoke(response)
print(answer)
```

项目里已经放了可直接运行的版本：

```bash
cd /Users/weijiaxin/Documents/pythonwork/gptbri
conda activate gptbri
python test_langchain_client.py
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-local-test",
    base_url="http://127.0.0.1:8011/v1",
)

resp = client.chat.completions.create(
    model="chatgpt-web",
    messages=[
        {"role": "user", "content": "你好，只回复一句话"}
    ],
)

print(resp.choices[0].message.content)
```

### curl

```bash
curl http://127.0.0.1:8011/v1/chat/completions \
  -H "Authorization: Bearer sk-local-test" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatgpt-web",
    "messages": [
      {"role": "user", "content": "只回复 ok"}
    ]
  }'
```

## Tool Calling

这个服务支持 OpenAI 风格的 `tools/tool_calls`，可以跑 LangChain 的 `bind_tools()` agent 流程。

直接运行项目里的测试：

```bash
cd /Users/weijiaxin/Documents/pythonwork/gptbri
conda activate gptbri
python test_tool_client.py
```

完整流程是：

```text
用户问题：请计算 2*3 是多少？
        ↓
ChatGPT 网页返回工具调用 JSON
        ↓
API 服务把 JSON 包装成 OpenAI 标准 tool_calls
        ↓
LangChain 在本地执行 multiply(a=2, b=3)
        ↓
工具结果 6 作为 ToolMessage 发回 API
        ↓
ChatGPT 根据工具结果生成最终回答
```

重点：工具函数始终在你本地 Python 里执行。ChatGPT 不会真的执行你的函数，它只负责判断是否需要调用工具、调用哪个工具、传什么参数，以及最后把工具结果组织成自然语言回答。

如果 ChatGPT 偶发没有输出合法 JSON，可以临时打开本地规则兜底：

```env
BRIDGE_USE_LOCAL_TOOL_FALLBACK=1
```

## 聊天框管理

服务默认只管理两个 ChatGPT 标签页：

```text
普通聊天请求 → chat 标签页 / chat 对话槽
工具调用请求 → tool 标签页 / tool 对话槽
```

这样普通聊天和工具调用互不污染，同时不会每问一次就新开一个页面。

启动时预开两个标签页：

```env
BRIDGE_PREOPEN_CONVERSATION_PAGES=1
```

启动时关闭多余的 ChatGPT 标签页：

```env
BRIDGE_CLOSE_EXTRA_CHATGPT_PAGES=1
```

每个聊天框使用多久后重开：

```env
BRIDGE_CHAT_RESET_SECONDS=1800
```

常用取值：

```text
0      永不按时间自动重开，始终复用同一个 ChatGPT 对话
300    5 分钟后重开
1800   30 分钟后重开
3600   1 小时后重开
```

这个计时按对话槽分别计算：`chat` 和 `tool` 各算各的。超过时间后，不会打断正在执行的请求，而是在对应槽的下一次请求开始时重开聊天。

## 图片输入

图片按 OpenAI 视觉格式传。`image_url.url` 支持：

```text
本地路径：/absolute/path/image.png
file URL：file:///absolute/path/image.png
HTTP URL：https://example.com/image.png
base64 data URL：data:image/png;base64,...
```

示例：

```python
resp = client.chat.completions.create(
    model="chatgpt-web",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请总结这张图"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "/absolute/path/image.png"
                    },
                },
            ],
        }
    ],
)

print(resp.choices[0].message.content)
```

## 配置说明

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_MODEL` | `chatgpt-web` | 客户端传给 LangChain/OpenAI SDK 的模型名 |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8011/v1` | 客户端请求地址 |
| `OPENAI_API_KEY` | `sk-local-test` | 客户端 API key，要和 `BRIDGE_API_KEY` 一致 |
| `BRIDGE_API_KEY` | 空 | 本地 API 服务校验的 Bearer token；为空则不校验 |
| `BRIDGE_MODEL_NAME` | `chatgpt-web` | 本地服务暴露出来的模型名 |
| `BRIDGE_HOST` | `127.0.0.1` | 本地 API 监听地址 |
| `BRIDGE_PORT` | `8000` | 本地 API 监听端口；本项目 `.env.example` 推荐用 `8011` |
| `BRIDGE_CDP_URL` | `http://127.0.0.1:9222` | Chrome 远程调试地址 |
| `BRIDGE_CHAT_URL` | `https://chatgpt.com/` | 新聊天入口 |
| `BRIDGE_REUSE_CDP_PAGE` | `1` | 复用已托管的 ChatGPT 标签页 |
| `BRIDGE_CLOSE_EXTRA_CHATGPT_PAGES` | `1` | 启动时关闭多余 ChatGPT 标签页 |
| `BRIDGE_PREOPEN_CONVERSATION_PAGES` | `1` | 启动时预开普通聊天和工具调用标签页 |
| `BRIDGE_CHAT_CONVERSATION_KEY` | `chat` | 普通聊天使用的对话槽名称 |
| `BRIDGE_TOOL_CONVERSATION_KEY` | `tool` | 工具调用使用的对话槽名称 |
| `BRIDGE_CHAT_RESET_SECONDS` | `0` | 每个对话槽多久后重开；`.env.example` 建议 `1800` |
| `BRIDGE_TIMEOUT_SECONDS` | `180` | 单次网页请求等待时间 |
| `BRIDGE_MAX_IMAGE_MB` | `25` | 单张图片大小上限 |
| `BRIDGE_USE_LOCAL_TOOL_FALLBACK` | `0` | 工具 JSON 规划失败时是否启用本地规则兜底 |

换端口时，需要同时改两处：

```env
OPENAI_BASE_URL=http://127.0.0.1:8020/v1
BRIDGE_PORT=8020
```

## API 接口

支持：

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

也提供无 `/v1` 的别名：

```text
GET  /models
POST /chat/completions
```

`stream=true` 可以请求，但底层网页无法做到真正 token 级流式输出。服务会等待完整回复生成后，再用 SSE 格式一次性返回，主要用于兼容已有客户端。

## 单脚本模式

如果不需要 OpenAI 兼容 API，也可以直接使用底层桥接脚本。

命令行：

```bash
python chatgpt_bridge.py \
  --cdp-url http://127.0.0.1:9222 \
  --text "请总结这张图" \
  --image /absolute/path/image.png
```

Python：

```python
import asyncio
from pathlib import Path
from chatgpt_bridge import ChatGPTWebBridge

async def main():
    async with ChatGPTWebBridge(cdp_url="http://127.0.0.1:9222") as bridge:
        reply = await bridge.ask(
            text="请识别图片内容并给出结构化摘要",
            images=[Path("/absolute/path/image.png")],
        )
        print(reply)

asyncio.run(main())
```

## 项目文件

```text
chatgpt_bridge.py        浏览器自动化核心：输入文本/图片并读取回复
llm_api_server.py        OpenAI 兼容 API 服务
launch_chrome_debug.sh   启动带调试端口的专用 Chrome
run_api_server.sh        启动本地 API 服务，会自动加载 .env
env_setup.py             示例脚本读取 .env 的工具
test_langchain_client.py LangChain 普通聊天调用示例
test_tool_client.py      LangChain bind_tools 工具调用示例
.env.example             配置模板
requirements.txt         Python 依赖
```

## 常见问题

### 服务连不上 Chrome

先确认 Chrome 调试端口在线：

```bash
curl http://127.0.0.1:9222/json/version
```

如果失败，重新启动：

```bash
bash launch_chrome_debug.sh
```

### ChatGPT 一直安全认证

不要让 Playwright 自动登录。推荐使用项目里的 Chrome 启动脚本：

```bash
bash launch_chrome_debug.sh
```

然后在打开的真实 Chrome 里手动登录 ChatGPT。登录成功、能看到输入框后，再启动 API 服务。

### API 打开了很多 ChatGPT 页面

确认 `.env` 里保留：

```env
BRIDGE_REUSE_CDP_PAGE=1
BRIDGE_CLOSE_EXTRA_CHATGPT_PAGES=1
BRIDGE_PREOPEN_CONVERSATION_PAGES=1
BRIDGE_CHAT_CONVERSATION_KEY=chat
BRIDGE_TOOL_CONVERSATION_KEY=tool
```

然后重启 API 服务。正常情况下只会保留普通聊天和工具调用两个托管标签页。

### 请求返回旧上下文或旧图片内容

确认使用新聊天入口：

```env
BRIDGE_CHAT_URL=https://chatgpt.com/
```

普通聊天和工具调用各自保存自己的 `/c/...` 会话 URL，不互相污染。除非你明确要继续某个旧会话，否则不要把 `BRIDGE_CHAT_URL` 配成具体的 `/c/...` 地址。

### 端口被占用

检查端口：

```bash
lsof -nP -iTCP:8011 -sTCP:LISTEN
```

换端口：

```env
OPENAI_BASE_URL=http://127.0.0.1:8020/v1
BRIDGE_PORT=8020
```

### LangChain 连接失败

先确认 `.env`：

```env
OPENAI_MODEL=chatgpt-web
OPENAI_BASE_URL=http://127.0.0.1:8011/v1
OPENAI_API_KEY=sk-local-test
```

再确认服务健康：

```bash
curl http://127.0.0.1:8011/health
```

### 没有校验 API key

`run_api_server.sh` 会自动加载 `.env`。如果没有设置 `BRIDGE_API_KEY`，服务允许无 key 调用。

建议保留：

```env
BRIDGE_API_KEY=sk-local-test
```

客户端使用：

```text
Authorization: Bearer sk-local-test
```

## 限制

- 不适合高并发：内部会串行排队操作同一个浏览器
- 不保证长期稳定：依赖 ChatGPT 网页结构和按钮选择器
- 不是真正流式：`stream=true` 只是兼容已有客户端
- 需要保持专用 Chrome 打开，并且 ChatGPT 已登录
- 可能受到 ChatGPT 网页安全认证、速率限制、账号状态影响
- 建议只绑定 `127.0.0.1` 本机使用，不要暴露到公网

## 适合的使用场景

- 本机快速把 ChatGPT 网页接进 LangChain 示例
- 测试 OpenAI 兼容客户端代码
- 调试支持 `tools/tool_calls` 的 agent 流程
- 临时验证文本和图片输入链路
- 在不改业务代码的情况下，把模型来源切到 ChatGPT 网页
