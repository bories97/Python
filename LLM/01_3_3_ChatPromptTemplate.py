# LangChain을 사용하여 ChatPromptTemplate을 생성하고, 이를 Google의 
# Gemini 모델과 연결하여 답변을 받아오는 전형적인 구조
### 2-튜플 형태의 메시지 목록으로 프롬프트 생성 (type, content)

# LangChain에서 대화형 프롬프트를 만들기 위한 기본 클래스인 ChatPromptTemplate을 불러옵니다.
from langchain_core.prompts import ChatPromptTemplate

# 시스템 메시지(AI의 역할 설정)와 사용자 메시지(질문)를 리스트 안에 튜플 (역할, 내용) 형태로 정의
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

# format_messages 메서드를 사용하여 {user_input}에 실제 질문을 대입합니다. 
# 결과적으로 LangChain 내부 객체 형태의 메시지 리스트가 생성됩니다.
messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
messages

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# ChatOllama는 기본적으로 GPU가 있으면 최우선으로 사용합니다.
llm = ChatOllama(
    model="gemma3:4b",  
    temperature=0,
    num_gpu=1,          # GPU 1개 사용 명시
)

chain = chat_prompt | llm | StrOutputParser()

# invoke 명령어로 체인을 실행
chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

### MessagePromptTemplate 활용
# 각 메시지 타입을 명시적인 클래스로 관리하므로, 구조가 복잡한 프로젝트에서 유지보수하기 좋습니다.

# 시스템용 메시지 템플릿과 사용자용 메시지 템플릿을 각각 별도의 클래스로 가져옵니다.
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# 튜플 대신 전용 클래스 메서드인 from_template을 사용해 프롬프트를 구성합니다. 
# 기능은 위와 동일하지만, 메시지별로 더 세부적인 설정을 넣을 때 유리합니다.
chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

# 실제 보낼 메시지 객체들을 생성
messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
messages

# 실행 가능한 체인을 만듭
chain = chat_prompt | llm | StrOutputParser()

chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

"""
첫 번째 방식은 빠르고 간편한 작성에 좋고, 두 번째 방식은 메시지 유형을 명확하게 구분하여 
관리할 때 좋습니다. 결과적으로 모델이 받는 데이터의 형태는 동일합니다!
"""
