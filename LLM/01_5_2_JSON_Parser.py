"""
01_5_2_JSON_Parser
Pydantic이라는 도구를 사용하여 AI의 답변 구조를 미리 설계하고, 그 구조에 맞춰 
결과를 JSON(파이썬 딕셔너리) 형태로 받아오는 코드입니다.
"""
### 데이터 구조 정의 (Pydantic 모델)

# 모델의 출력을 JSON 형식으로 파싱(해석)하는 도구를 가져옵니다.
from langchain_core.output_parsers import JsonOutputParser
# 데이터의 틀을 정의하고 각 항목에 설명을 달기 위한 도구를 가져옵니다.
from pydantic import BaseModel, Field

# 자료구조 정의 (pydantic)
class CusineRecipe(BaseModel): # CusineRecipe라는 이름의 데이터 양식을 정의
    # name이라는 항목은 문자열(str)이어야 하며, 요리의 이름이 들어갈 자리라고 AI에게 알려줍니다.
    name: str = Field(description="요리의 이름")
    # recipe 항목 역시 문자열이며, 상세 조리법이 들어갈 자리임을 명시합니다.
    recipe: str = Field(description="요리법 (조리 순서 포함)")

# 출력 파서 정의. 양식에 맞춰 답변을 해석할 파서를 생성
output_parser = JsonOutputParser(pydantic_object=CusineRecipe)

# AI가 JSON 형식을 잘 지킬 수 있도록 돕는 자동 생성된 지시 문구를 가져옵니다. 
# (내부적으로는 "출력은 반드시 { ... } 형태여야 한다"는 내용이 담깁니다.)
format_instructions = output_parser.get_format_instructions()

print(format_instructions) # 화면에 출력

# 프롬프트 템플릿 도구를 가져옵니다.
from langchain_core.prompts import PromptTemplate

# prompt 구성
prompt = PromptTemplate( # AI에게 보낼 전체 메시지 틀
    # "한국어로 답하세요"라는 명령과 함께, 위에서 만든 형식 지시 사항(format_instructions)과 
    # 사용자의 실제 질문(query)을 순서대로 배치합니다.
    template="Answer the user query in Korean..\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

print(prompt)

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

# temperature=0으로 설정하여 답변이 매번 바뀌지 않고 정확하게 형식을 지키도록 합니다.
llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# [프롬프트 구성 → 모델 답변 → JSON 해석] 과정을 하나의 파이프라인(체인)으로 연결합니다.
chain = prompt | llm | output_parser

# 결과값은 문자열이 아니라, 파이썬에서 즉시 사용할 수 있는 딕셔너리 객체가 됩니다.
result = chain.invoke({"query": "비빔밥 만드는 법 알려줘"})
print(result)