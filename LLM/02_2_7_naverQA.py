"""
02_2_7_naverQA
네이버 뉴스 기사를 긁어와서(Crawling), 내용을 잘게 나누고(Chunking), 이를 벡터 
저장소에 저장한 뒤, 질문에 대해 관련 있는 내용만 찾아 답변하는 
RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템의 전 과정을 담고 있습니다.
"""
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# 웹 사이트 접속 시 식별을 위한 사용자 에이전트 이름 설정
os.environ["USER_AGENT"] = "MyLangchainApp/1.0"

# pip install -U langchain langchainhub --quiet

import bs4
from langchain_classic import hub  # 프롬프트(Prompt)**를 저장해둔 도서관
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

# prompt = hub.pull("rlm/rag-prompt") 카드등록해야함.
# print("성공적으로 가져왔습니다!")

# # 1. 프롬프트 만들기
# my_prompt = PromptTemplate.from_template("너는 최고의 요리사야. {food} 레시피를 알려줘.")

# import os
# # pip install python-dotenv
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("LANGSMITH_API_KEY")

# 2. 허브에 업로드 (사용자명/저장소명 형식)
hub.push("kmdadoo/my-prompt", my_prompt)

# BeautifulSoup의 SoupStrainer를 사용해 특정 태그(div)와 클래스명만 필터링하도록 설정합니다.
# 뉴스 본문과 제목이 들어있는 부분만 골라내기 위함입니다.
bs4.SoupStrainer(
    "div",   # 태그 이름이 "div"인 요소만 고려해야 함
    #  "div" 요소를 클래스 속성에 따라 추가로 필터링한다. "newsct_article _article_body" 또는 "media_end_head_title" 값을 갖는 클래스 속성이 있는 "div" 요소만 선택
    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
)

# 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",), # 뉴스기사 링크
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)

docs = loader.load() # 실제 데이터를 가져와 docs 변수에 저장
print(f"문서의 수: {len(docs)}")
docs

# 각 텍스트 청크가 최대 1000자여야 함을 지정, 각 청크는 이전 청크와 100자씩 겹칩니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # 객체 생성

# 문서를 조각(Chunk)들로 나눕니다.
splits = text_splitter.split_documents(docs) # 문서 분할
len(splits) # 메서드의 의해 생성된 청구의 수

# 한국어 성능이 좋은 BAAI/bge-m3 모델을 사용하여 문장을 벡터로 변환합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# 분할된 텍스트 조각들을 벡터 스토어(FAISS)에 저장합니다. 이제 질문과 유사한 조각을 찾을 수 있습니다.
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings_model)

# 뉴스에 포함되어 있는 정보를 검색하고 생성합니다. 벡터 스토어를 검색기(retriever)로 변환합니다.
retriever = vectorstore.as_retriever()

from langchain_core.prompts import PromptTemplate

# AI가 어떻게 답변해야 할지 지침을 담은 템플릿을 만듭니다.
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
)

from langchain_google_genai import ChatGoogleGenerativeAI

# 구글의 Gemini 모델을 불러옵니다. temperature=0은 답변의 일관성을 위해 창의성을 최소화하는 설정입니다.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 파이프라인(Chain) 구성:
# 1. 질문을 받아 retriever에서 관련 문맥(context)을 찾음
# 2. 질문과 문맥을 프롬프트에 넣음
# 3. 모델(llm)에 전달하여 답변 생성
# 4. 출력 결과를 문자열로 파싱
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# pip install langchain_teddynote

# 테디노트 라이브러리를 사용하여 실시간으로 답변이 출력(Stream)되도록 실행합니다.
from langchain_teddynote.messages import stream_response

# 질문 예시들
answer = rag_chain.stream("부영그룹의 출산 장려 정책에 대해 설명해주세요.")
stream_response(answer)

answer = rag_chain.stream("부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?")
stream_response(answer)

answer = rag_chain.stream("정부의 저출생 대책을 bullet points 형식으로 작성해 주세요.")
stream_response(answer)

answer = rag_chain.stream("부영그룹의 임직원 숫자는 몇명인가요?")
stream_response(answer)