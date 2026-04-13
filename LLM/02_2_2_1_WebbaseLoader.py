"""
02_2_2_1_WebBaseLoader
웹 사이트에서 필요한 정보만 똑똑하게 골라내어 데이터로 만드는 과정입니다.
"""
# 웹페이지의 HTML 소스를 분석하고 원하는 태그를 찾아주는 도구인 BeautifulSoup4를 가져옵니다.
import bs4
# 웹 URL에서 데이터를 읽어와서 LangChain에서 사용할 수 있는 문서 객체(Document) 형태로 변환해주는 로더를 가져옵니다.
from langchain_community.document_loaders import WebBaseLoader

# 여러 개의 url 지정 가능
# 데이터를 수집할 블로그 주소를 변수에 저장합니다. 여기서는 Replit 고객 사례와 LangGraph 업데이트 소식을 담은 블로그입니다.
url1 = "https://blog.langchain.dev/customers-replit/"
url2 = "https://blog.langchain.dev/langgraph-v0-2/"

# 로더(Loader) 설정 : 가장 핵심적인 부분으로, 데이터를 어떻게 가져올지 규칙을 정합니다.
loader = WebBaseLoader( # 웹 로더를 초기화
    web_paths=(url1, url2), # 한 번에 여러 개의 페이지를 읽어오도록 경로를 지정
    bs_kwargs=dict( # BeautifulSoup에 전달할 옵션을 설정
         # 매우 중요한 부분입니다. 웹페이지 전체를 다 가져오면 광고나 메뉴 같은 불필요한 정보가 섞입니다.
        parse_only=bs4.SoupStrainer(
            # HTML 태그 중 클래스 이름이 article-header(제목 부분)와 article-content(본문 내용)인 
            # 요소만 쏙 골라내어 가져오도록 필터링합니다.
            class_=("article-header", "article-content")
        )
    ),
)

# 설정한 규칙에 따라 웹페이지에 접속하여 데이터를 긁어온 뒤, docs라는 리스트에 저장합니다.
docs = loader.load()
# 로드된 문서의 개수를 확인합니다. (여기서는 url1, url2 두 개를 넣었으므로 결과는 2가 나옵니다.)
len(docs)

# 리스트의 첫 번째 항목, 즉 url1 (Replit 블로그)에서 가져온 데이터를 담고 있는 Document 객체를 보여줍니다.
docs[0]

# page_content: 아까 필터링한 제목과 본문의 실제 텍스트 내용입니다.
docs[0].page_content

# metadata 속성: 이 데이터가 어디서 왔는지(source: URL)에 대한 정보입니다.
docs[0].metadata


### 위에 docs[0] 한글로 번역하기. 올라마 활용

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 번역을 위한 모델 설정
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatOllama(model="gemma3:4b", temperature=0)

# 2. 번역용 프롬프트 구성
translation_prompt = ChatPromptTemplate.from_template(
    "다음은 웹페이지에서 가져온 영어 본문입니다. 개발자의 관점에서 매끄러운 한글로 번역해주세요:\n\n{content}"
)

# 3. 번역 체인 생성
translation_chain = translation_prompt | llm | StrOutputParser()

# 4. docs[0]의 본문을 번역 실행
# docs[0].page_content는 앞서 WebBaseLoader로 가져온 영어 텍스트입니다.
korean_result = translation_chain.invoke({"content": docs[0].page_content})

print(korean_result)

