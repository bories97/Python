# LangChain의 프롬프트 템플릿(Prompt Template) 관리 방법과 여러 템플릿을 하나로 합치는 
# 기법, 그리고 이를 모델에 연결하여 실행하는 전 과정을 보겠습니다.

### 1.기본 프롬프트 템플릿 생성
# 프롬프트(AI에게 줄 질문)의 틀을 만드는 도구를 가져옵니다.
from langchain_core.prompts import PromptTemplate

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
# 위 문자열을 LangChain이 관리할 수 있는 '템플릿 객체'로 변환합니다.
prompt_template = PromptTemplate.from_template(template_text)

# 템플릿에 값을 채워서 프롬프트를 완성
# format 함수를 사용해 {name}에는 "홍길동", {age}에는 30을 실제로 채워 넣습니다.
filled_prompt = prompt_template.format(name="홍길동", age=30)

# 문자열 출력
filled_prompt

### 2.프롬프트 템플릿 결합 (중요)
# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
# 결과적으로 {name}, {age}, {language} 세 개의 변수를 가지는 하나의 커다란 템플릿이 됩니다.
combined_prompt = (
    prompt_template    # 앞서 만든 "안녕하세요..." 템플릿
    # 새로운 문장이 담긴 템플릿을 더합니다.
    + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
    # 마지막으로 번역 언어를 지정하는 일반 문자열을 더합니다.
    + "\n\n{language}로 번역해주세요." 
)

# 합쳐진 전체 템플릿 구조를 확인합니다.
combined_prompt

### 3.마크다운 출력 확인 (디버깅용)
# pip install ipython
# 마크다운 형식으로 데이터를 보여주기 위한 도구를 가져옵니다.
from IPython.display import Markdown, display

# 세 가지 변수를 모두 채운 최종 문장을 마크다운 객체로 만듭니다.
result = Markdown(combined_prompt.format(name="홍길동", age=30, language="영어"))
# 터미널 환경이므로 객체 내용(텍스트 데이터)만 출력합니다.
print(result.data) 

#display(result) # 쥬피터 노트북용

### 4. 모델 설정 및 체인 실행
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.getenv("GOOGLE_API_KEY")

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# ChatOllama는 기본적으로 GPU가 있으면 최우선으로 사용합니다.
llm = ChatOllama(
    model="gemma3:4b",  # 설치된 정확한 모델명 확인 (gemma3가 없다면 gemma2 추천)
    temperature=0,
    # 필요하다면 추가적인 GPU 설정을 넣을 수 있습니다.
    num_gpu=1,          # GPU 1개 사용 명시
)  # 구글 오픈 모델
# [프롬프트 -> AI 모델 -> 텍스트 추출] 과정을 파이프로 연결하여 '체인'을 만듭니다.
chain = combined_prompt | llm | StrOutputParser()
# 딕셔너리 형태로 필요한 모든 값을 전달하여 전체 체인을 실행합니다.
result = chain.invoke({"age":30, "language":"영어", "name":"홍길동"})
result