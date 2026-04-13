### Groq를 이용한 고속 추론 및 한국 관련 질의
# pip install -U langchain-groq
# pip install groq

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# Groq 사용을 위한 라이브러리 로드
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ChatGroq 모델 초기화
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,                 # 답변의 창의성 조절 (0.7은 약간의 다양성을 허용함)
    max_tokens=300,                  # 답변의 최대 길이를 300토큰으로 제한
    api_key=GROQ_API_KEY
)

# 시스템 메시지(AI의 정체성)와 사용자 질문이 결합된 프롬프트 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절하고 유익한 AI 조수입니다. 한국의 역사와 문화에 대해 잘 알고 있습니다."),
    ("human", "{question}")
])

# Chain 생성. 프롬프트와 Groq LLM을 연결하여 체인 생성
chain = prompt | llm

# 반복문을 돌리기 위한 질문 리스트 정의
questions = [
    "한글의 창제 원리는 무엇인가요?",
    "김치의 역사와 문화적 중요성에 대해 설명해주세요.",
    "조선시대의 과거 제도에 대해 간단히 설명해주세요.",
]

# 각 질문에 대한 답변 생성
for question in questions:
    response = chain.invoke({"question": question}) # 체인 실행
    print(f"질문: {question}")
    print(f"답변: {response.content}\n") # ChatGroq 결과는 .content에 들어있습니다.
    print("-" * 30 + "\n")

# pip list --format=freeze > requirements.txt
# pip install -r requirements.txt