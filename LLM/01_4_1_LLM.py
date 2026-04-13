"""
LangChain을 사용하여 Google Gemini 모델을 설정하고, 이를 단순히 호출하는 방법과 
**프롬프트 템플릿을 결합하여 구조화된 체인(Chain)**으로 만드는 방법을 보여줍니다.
"""
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
### 1. 기본 모델 호출 방식
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 모델을 초기화합니다. 모델명은 "gemini-2.5-flash"를 사용합니다.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 모델에 직접 문자열 질문을 던지고 답변을 받습니다. 
# invoke는 '실행하다/호출하다'라는 뜻으로 LangChain의 표준 호출 메서드입니다.
llm.invoke("한국의 대표적인 관광지 3군데를 추천해주세요.")

### 2. 프롬프트 템플릿과 체인(Chain) 활용 방식

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# 사용할 채팅 모델을 다시 선언합니다.
chat = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 대화의 형식을 정의하는 '프롬프트 템플릿'을 만듭니다.
chat_prompt = ChatPromptTemplate.from_messages([
    # 시스템 메시지: AI에게 '여행 전문가'라는 정체성을 부여하여 답변의 전문성을 높입니다.
    ("system", "이 시스템은 여행 전문가입니다."),
    # 유저 메시지: 사용자가 입력할 내용이 들어갈 자리({user_input})를 변수로 지정합니다.
    ("user", "{user_input}"),
])

# 파이프 기호(|)를 사용하여 프롬프트와 모델을 하나로 엮습니다. (LCEL 기법)
# 이제 이 'chain'은 입력을 받으면 프롬프트 형식에 맞춘 뒤 모델에 전달합니다.
chain = chat_prompt | chat

# 사용자의 입력을 딕셔너리 형태로 전달하여 전체 프로세스를 실행합니다.
chain.invoke({"user_input": "안녕하세요? 한국의 대표적인 관광지 3군데를 추천해주세요."})