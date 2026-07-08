from env_setup import load_project_env

load_project_env()

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

output_parser = StrOutputParser()
answer = output_parser.invoke(response)
print(answer)
