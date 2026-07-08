from env_setup import load_project_env

load_project_env()

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


@tool
def add(a: int, b: int) -> int:
    """计算两个整数的加法 a + b。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘法 a * b。"""
    return a * b


tools = [add, multiply]

model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
    temperature=0.7,
)

llm_with_tool = model.bind_tools(tools)

query = "请计算2*3是多少？"
messages = [HumanMessage(content=query)]

ai_message = llm_with_tool.invoke(messages)
print("模型工具调用:", ai_message.tool_calls)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print(f"工具执行结果:{tool_msg}")
    messages.append(tool_msg)

print("完整消息上下文:", messages)

result = llm_with_tool.invoke(messages)
print(f"AI最终回复:{result.content}")
