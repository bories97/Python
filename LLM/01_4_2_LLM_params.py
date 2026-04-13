"""
01_4_2_LLM_params
LangChain 라이브러리를 사용하여 Google의 Gemini 모델을 설정하고 호출하는 다양한 방법
"""
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

# 모델 파라미터 설정. 모델의 기본 동작(온도, 출력 길이 등)을 설정하는 딕셔너리
params = { 
    # 답변의 창의성을 조절합니다. (0에 가까우면 일관적, 1에 가까우면 창의적)
    "temperature": 0.7,         # 생성된 텍스트의 다양성 조정
    # 생성할 최대 단어(토큰) 수를 300개로 제한
    "max_output_tokens": 300,          # 생성할 최대 토큰 수
}

# 텍스트 생성 시 단어 선택의 세부 규칙을 정합니다.
kwargs = {
    # 똑같은 단어를 반복해서 사용하는 것을 방지
    "frequency_penalty": 0.5,   # 이미 등장한 단어의 재등장 확률
    # 이미 다룬 내용 외에 새로운 주제/단어를 도입하도록 유도
    "presence_penalty": 0.5,    # 새로운 단어의 도입을 장려
}

# 모델 인스턴스를 생성할 때 설정
# 위에서 정의한 설정값들을 적용하여 실제 Gemini 모델 인스턴스를 만듭니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", **params, model_kwargs = kwargs)

# 모델 호출
question = "태양계에서 가장 큰 행성은 무엇인가요?" # 모델에게 물어볼 질문 문자열
response = model.invoke(input=question, stop=["\n", "."]) # 질문을 모델에 전달하고 응답을 받습니다.

# 전체 응답 출력
print(response) # 모델의 응답 객체 전체(메타데이터 포함)를 출력

### 간결한 모델 호출 및 내용 출력

# 모델 파라미터 설정
# 모델 생성 시 바로 파라미터를 인자로 전달합니다. 응답 길이를 50으로 더 짧게 잡았습니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_tokens= 50)
question = "태양계에서 가장 큰 행성은 무엇인가요?"

# 모델 인스턴스를 호출할 때 전달
response = model.invoke(input=question, stop=["\n", "."]) # 모델을 실행

# 문자열 출력
# 응답 객체 중에서 부가 정보를 제외한 **순수 답변 내용(텍스트)**만 출력합니다.
print(response.content)

### 프롬프트 템플릿 사용 (ChatPromptTemplate)

# 대화형 프롬프트 구조를 만들기 위한 도구를 불러옵니다.
from langchain_core.prompts import ChatPromptTemplate

# 모델의 역할(System)과 사용자의 질문(User) 형식을 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."), # 역할을 부여
    ("user", "{user_input}"), # 실제 질문이 들어갈 자리
])

# 제 질문을 넣어서 모델이 이해할 수 있는 메시지 리스트 형식으로 변환합니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=100)

messages = prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

# 템플릿이 적용된 메시지를 모델에 보내고 답변을 받습니다.
before_answer = model.invoke(messages, stop=["\n", "."])

# binding 이전 출력
print(before_answer)

### 체인(Chain) 구성 및 실행
# 모델 호출 시 추가적인 인수를 전달하기 (응답의 최대 길이를 30 토큰으로 제한(입출력 같이 계산))
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", max_tokens=30)
# 프롬프트와 모델을 연결합니다. 이제 chain을 호출하면 자동으로 프롬프트 형식이 적용된 후 모델로 전달됩니다.
chain = prompt | model

# 딕셔너리 형태로 질문을 넣으면 체인이 작동하여 최종 답변을 가져옵니다.
after_answer = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"}, stop=["\n", "."])

# 출력
print(after_answer) # 체인을 통해 생성된 짧은 답변을 출력