"""
04_LangChain_Agent
LangChain의 에이전트(Agent) 기능을 사용하여 외부 도구(웹 검색 및 파이썬 코드 실행)를 LLM이 
스스로 판단하여 사용하게 만드는 고급 구성 예시
"""
# pip install tavily-python

from langchain_community.tools import TavilySearchResults

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()

# TAVILY API KEY를 기입합니다.
os.environ["TAVILY_API_KEY"]

query = "2025년 애플의 주가 전망에 대해서 분석하세요."

# pip install -U langchain-tavily

web_search = TavilySearchResults(max_results=2)

search_results = web_search.invoke(query)

for result in search_results:
    print(result)
    print("-" * 100)
    
# pip install langgraph
## 웹 검색 도구 설정 (Tavily Search)
from langchain_experimental.tools import PythonAstREPLTool   #  Python 코드를 안전하게 실행하기 위한 도구
from langgraph.prebuilt import create_react_agent # 에이전트 초기화 및 타입 설정
from langchain_google_genai import ChatGoogleGenerativeAI # 구글AI의 채팅 모델 사용을 위한 클래스

# 2. 도구 및 모델 설정
# PythonAstREPLTool을 사용하여 Python 코드를 실행할 수 있는 환경을 생성합니다.
tools = [PythonAstREPLTool()]

# OpenAI의 GPT-4 모델 초기화. temperature=0은 가장 확실한 응답을 얻기 위한 설정입니다.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. 프롬프트 가져오기
prompt = "당신은 파이썬 비서입니다. 답변 시 반드시 사용한 코드를 마크다운 형식으로 포함하세요. 모든 설명은 한국어로 합니다."

# 4. LangGraph 에이전트 생성
# LangGraph의 create_react_agent는 'model'과 'tools'가 필수 인자입니다.
agent = create_react_agent(llm, tools)

# 에이전트에게 작업 지시 및 실행
query = """
1부터 10까지의 숫자 중 짝수만 출력하는 Python 코드를 작성하고 실행해주세요.
답변할 때 실행한 '전체 파이썬 코드'와 '실행 결과'를 모두 포함해서 설명해주세요.
"""

result = agent.invoke({
    "messages": [
        ("system", prompt), 
        ("user", query)
    ]
})

# 결과 출력 : 에이전트의 작업 결과를 출력합니다.
print(result)

# 가장 마지막 메시지의 내용만 가져오기
final_answer = result["messages"][-1].content
print(final_answer)
