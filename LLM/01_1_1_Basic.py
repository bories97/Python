# pip install langchain tiktoken langchain-google-genai 
"""
langchain: LLM(거대언어모델)을 이용한 애플리케이션을 쉽게 만들 수 있도록 도와주는 프레임워크입니다. 모델 연결, 기억 저장 등을 편리하게 해줍니다.
tiktoken: 텍스트를 AI가 이해할 수 있는 단위인 '토큰'으로 쪼개주는 도구입니다. 주로 비용 계산이나 글자 수 제한 확인용으로 쓰입니다.
langchain-google-genai: Google의 Gemini 모델을 LangChain에서 사용할 수 있게 연결해주는 전용 어댑터입니다.
--quiet: 설치 과정 중에 나오는 복잡한 메시지들을 생략하고 조용히 설치하라는 옵션입니다.
"""

# 내 컴퓨터의 시스템 운영체제(Windows, Mac 등) 정보를 다루는 파이썬 기본 도구를 가져옵니다. 
# 주로 파일 경로 설정이나 비밀번호(API 키)를 관리할 때 사용합니다.
import os
# 설치한 패키지 중에서 Gemini 모델과 대화하기 위해 필요한 핵심 클래스(기능) 하나만 골라서 가져오는 것입니다.
from langchain_google_genai import ChatGoogleGenerativeAI
# os.environ: 운영체제가 기억하고 있는 '설정값 리스트'라고 생각하시면 됩니다
os.environ["GOOGLE_API_KEY"] = "제미나이 API"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# llm.invoke(...): AI에게 질문을 던지는(호출하는) 핵심 명령어입니다. 괄호 안의 텍스트를 입력으로 보냅니다.
result = llm.invoke("안녕! 너에대해 짧게 소개해줘.")
# result.content: AI의 응답 중 순수한 답변 텍스트 내용만 뽑아내겠다는 뜻입니다.
print(result.content)