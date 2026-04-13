"""
01_2_3_Runnable
**LCEL(LangChain Expression Language)**을 활용하여 AI 모델을 호출하고, 
다양한 방식(단일 실행, 일괄 실행, 실시간 스트리밍, 비동기 실행)으로 데이터를 처리하는 예제
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 컴포넌트 정의
# {topic} 자리에 원하는 단어를 넣을 수 있는 질문 템플릿을 만듭니다.
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
output_parser = StrOutputParser()

# 2. 체인 생성
chain = prompt | model | output_parser

# 3. invoke 메소드 사용
result = chain.invoke({"topic":"지구 자전"})
print("invoke 결과 : ", result)

### 일괄 처리 (Batch)
# batch 메소드 사용
# 여러 개의 주제를 리스트로 준비합니다.
topics = ["지구 공전", "화산 활동", "대륙 이동"]
# 여러 질문을 동시에(혹은 효율적으로 순차적으로) 처리하여 리스트로 결과를 받습니다.
results = chain.batch([{"topic": t} for t in topics])
for topic, result in zip(topics, results):
    # 각 주제별로 결과의 앞부분 50자만 출력합니다.
    print(f"{topic} 설명: {result[:50]}...")
    
### 실시간 스트리밍 (Stream)
# stream 메소드 사용
# 답변이 완성될 때까지 기다리지 않고, 생성되는 대로 한 글자씩 가져옵니다.
stream = chain.stream({"topic": "지진"}) 
print("stream 결과:")
for chunk in stream:
    # AI가 답변을 생성하는 즉시 화면에 출력하여 실시간처럼 보이게 합니다.
    print(chunk, end="", flush=True)
print()

### 비동기 실행 환경 설정 (nest_asyncio)
# pip install nest_asyncio
# 파이썬의 비동기 처리(async)를 돕는 라이브러리들을 가져옵니다.
import nest_asyncio
import asyncio

# nest_asyncio 적용 (구글 코랩 등 주피터 노트북에서 실행 필요)
# 주피터 노트북이나 특정 환경에서 '이미 실행 중인 이벤트 루프'와 충돌하지 않게 설정합니다.
nest_asyncio.apply()

### 비동기 실행 (ainvoke)
# 비동기 메소드 사용 (async/await 구문 필요)
# 비동기 방식으로 체인을 실행하는 함수를 정의합니다.
async def run_async():
    # await를 사용해 답변이 올 때까지 기다리는 동안 다른 작업을 할 수 있게 합니다.
    result = await chain.ainvoke({"topic": "해류"})
    print("ainvoke 결과:", result[:50], "...")

asyncio.run(run_async()) # 정의한 비동기 함수를 실제로 실행합니다.