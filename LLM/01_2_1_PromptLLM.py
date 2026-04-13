# LangChain의 핵심 문법인 **LCEL(LangChain Expression Language)**을 사용하여 
# 프롬프트 + 모델 + 출력 파서를 하나로 연결하는 완성된 흐름을 보여주겠습니다.
from langchain_google_genai import ChatGoogleGenerativeAI

# model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# chain 실행
result = llm.invoke("지구의 정확한 자전 주기는 100자로 설명해줘?")
print(result.content)

# ChatPromptTemplate: 사용자의 입력을 받아서 미리 정해진 양식(전문가 역할 부여 등)
# 으로 만들어주는 '편지 봉투' 역할을 가져옵니다.
from langchain_core.prompts import ChatPromptTemplate

# 역할 부여: AI에게 "너는 천문학 전문가야"라는 정체성을 부여하여 답변의 전문성을 높입니다.
# 변수 설정: {input} 부분은 나중에 사용자가 실제로 던질 질문이 들어갈 '빈칸'입니다.
# prompt = ChatPromptTemplate.from_template(
#     "당신은 천문학 전문가입니다. 다음 질문에 대해 전문적인 지식을 바탕으로 답변해 주세요. "
#     "<질문>: {input}"
# )
prompt = ChatPromptTemplate.from_template(
    "당신은 천문학 분야의 권위 있는 전문가입니다. "
    "사용자의 질문에 대해 고등학생도 이해하기 쉽게 자세하고 친절하게 설명해 주세요. "
    "항상 정확한 과학적 사실을 바탕으로 답변해야 합니다.\n\n"
    "<질문>: {input}\n"
    "<답변>:"
)
prompt

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# chain 연결 (LCEL)
chain = prompt | llm

# chain 호출
chain.invoke({"input":"지구의 정확한 자전 주기는 100자로 설명해줘?"})

from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser: AI의 복잡한 응답 결과 중에서 우리에게 필요한 
# **'문자열(결과 텍스트)'**만 깔끔하게 뽑아주는 '여과기' 역할을 가져옵니다.
from langchain_core.output_parsers import StrOutputParser

# prompt + model + output parser
prompt = ChatPromptTemplate.from_template(
    "당신은 천문학 전문가입니다. 다음 질문에 대해 전문적인 지식을 바탕으로 답변해 주세요. "
    "<질문>: {input}"
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 응답 정제: AI는 보통 답변 외에도 메타데이터(토큰 수, 모델명 등)를 함께 보냅니다. 
# 이 도구는 그중에서 우리가 읽을 수 있는 텍스트 내용만 남기고 나머지는 버리는 역할을 합니다.
output_parser = StrOutputParser()

# LCEL chaining(체인 연결) : | (파이프) 기호를 사용하여 데이터의 흐름을 한 줄로 연결
# prompt: 사용자의 입력을 받아 전문가 모드의 문장으로 변환합니다.
# llm: 변환된 문장을 AI에게 보내 답변을 생성합니다.
# output_parser: 생성된 복잡한 응답에서 답변 텍스트만 추출합니다.
chain = prompt | llm | output_parser

# chain 호출
# invoke: "체인을 가동하라"는 명령어입니다.
# {"input": "..."}: 템플릿의 {input} 빈칸에 들어갈 내용을 딕셔너리 형태로 전달합니다.
chain.invoke({"input":"지구의 정확한 자전 주기는 100자로 설명해줘?"})
