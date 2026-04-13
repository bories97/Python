"""
01_2_2_MultiChain
**LangChain(랭체인)**을 사용하여 두 개의 작업을 하나로 연결(Chain)하고, 
한국어 단어를 영어로 번역한 뒤 그 영어 단어의 뜻을 다시 한국어로 설명하는 구조
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# 첫 번째 질문 양식: "{korean_word}" 자리에 단어를 넣으면 영어로 번역하라고 명령합니다.
prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
# 두 번째 질문 양식: "{english_word}" 자리에 올 영어 단어를 옥스퍼드 사전을 참고해 한국어로 설명하라고 명령합니다.
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#  첫 번째 실행 체인
chain1 = prompt1 | llm | StrOutputParser()

# 실제로 "미래"라는 단어를 넣어 첫 번째 체인을 실행합니다. (결과: "Future")
chain1.invoke({"korean_word":"미래"})

chain2 = (
    {"english_word":chain1} # chain1의 결과값(번역된 영어 단어)을 "english_word"라는 변수에 담습니다.
    | prompt2  # 그 영어 단어를 prompt2(설명 요청)의 {english_word} 자리에 집어넣습니다.
    | llm    # 완성된 질문을 다시 AI 모델에게 보냅니다.
    | StrOutputParser()   # 최종 설명 결과에서 텍스트만 추출합니다.
)

# "미래"라는 입력값으로 전체 과정을 시작합니다.
chain2.invoke({"korean_word":"미래"})
# 과정: "미래" 입력 -> "Future"로 번역 -> "Future"에 대한 사전적 설명 생성 -> 최종 답변 출력