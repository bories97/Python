"""
01_5_1_CSV_Parser
AI의 답변을 쉼표(,)로 구분된 리스트 형태로 강제하고, 그 결과를 
파이썬의 리스트(List) 객체로 바로 변환해주는 과정을 담고 있습니다.
"""
### 출력 파서(Output Parser) 설정

# 쉼표로 구분된 텍스트를 파이썬 리스트로 바꿔주는 도구를 가져옵니다.
from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser() # 파서 인스턴스를 생성

# AI에게 "답변을 쉼표로 구분해서 작성해달라"고 요청할 때 사용할 지시 사항(가이드라인) 문구를 자동으로 생성합니다.
format_instructions = output_parser.get_format_instructions()

print(format_instructions) # 생성된 지시 문구를 출력합니다.

### 프롬프트 템플릿(Prompt Template) 구성

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# 프롬프트 틀을 만들기 위한 클래스를 가져옵니다.
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 프롬프트의 구조를 정의
prompt = PromptTemplate(
    # 실제 질문 내용입니다. {subject}에는 주제가, {format_instructions}에는 
    # "쉼표로 구분해달라"는 지시가 들어갑니다.
    template="List five {subject}.\n{format_instructions}", 
    input_variables=["subject"], # 나중에 사용자가 입력할 변수 이름을 지정
    # 변하지 않는 지시 사항 문구를 미리 템플릿에 채워넣습니다.
    partial_variables={"format_instructions": format_instructions},
)

### 모델 및 체인(Chain) 생성

# 리스트를 정확하게 뽑아야 하므로 창의성을 0으로 설정하여 일관된 답변을 유도
llm =  ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# [프롬프트 생성 → 모델 전달 → 결과 파싱] 과정을 하나로 묶습니다.
chain = prompt | llm | output_parser

# "인기 있는 한국 요리"라는 주제를 넣어 체인을 실행
chain.invoke({"subject": "popular Korean cusine"})