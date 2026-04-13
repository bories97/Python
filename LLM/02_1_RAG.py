"""
02_1_RAG
웹 페이지의 데이터를 가져와서 분석하고, 질문에 답하는 
RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템의 전체 과정을 담고 있습니다.
"""
### 1. 데이터 로드(Load Data)

# Data Loader - 웹페이지 데이터 가져오기
# 웹페이지 내용을 불러오는 도구를 가져옵니다.
from langchain_community.document_loaders import WebBaseLoader

# 위키피디아 정책과 지침. 위키피디아 주소(https://ko.wikipedia.org/wiki/위키백과:정책과_지침)
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
# 해당 주소에서 데이터를 가져올 로더를 생성
loader = WebBaseLoader(url)

# pip install beautifulsoup4
# 웹페이지를 실행해서 텍스트 데이터를 Documents 객체 리스트로 저장
docs = loader.load()

print(len(docs)) 
print(len(docs[0].page_content)) # 불러온 전체 글자 수를 확인
print(docs[0].page_content[5000:6000])

### 2. 텍스트 분할(Text Split)
# 너무 긴 텍스트를 AI가 처리하기 좋은 크기로 쪼개는 단계
# pip install -U langchain-text-splitters

# 문서 분할 (커다란 문서 객체들을 -> 작은 단위의 조각 문서들로 분할)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 한 조각당 1,000자로 자르되, 앞뒤 맥락 연결을 위해 200자를 겹치게 설정합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)

# 원본 문서를 쪼개서 조각 문서(splits) 리스트를 만듭니다.
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

# page_content 속성
# 11번째 조각의 본문 내용을 확인
splits[10].page_content

# metadata 속성
# 해당 조각이 어느 URL에서 왔는지 등의 부가 정보를 확인
splits[10].metadata

### 3. 인덱싱(Indexing)
# 텍스트를 숫자로 변환(임베딩)하여 데이터베이스에 저장하는 단계

# Indexing (Texts -> Embedding -> Store)
# 데이터를 저장하고 검색할 수 있는 '벡터 저장소'인 Chroma를 가져옵니다.
from langchain_community.vectorstores import Chroma
# 문장을 숫자로 변환하는 모델을 가져옵니다.
from langchain_community.embeddings import HuggingFaceEmbeddings

# 다국어 성능이 뛰어난 BGE-M3 모델을 사용하여 텍스트를 숫자로 수치화합니다.
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

# 쪼개진 텍스트 조각들을 숫자(벡터)로 변환하여 데이터베이스에 저장합니다.
vectorstore = Chroma.from_documents(splits, embedding=embeddings)

# 질문과 가장 비슷한 내용의 조각을 데이터베이스에서 찾아옵니다.
docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
print(len(docs))
print(docs[0].page_content)

### 생성(Generation)

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
# AI에게 전달할 지시문입니다. "제공된 맥락(Context) 안에서만 질문에 답하라"고 제한을 둡니다.
template = '''Answer the question based only on the following context:
    {context}

    Question: {question}
'''

# 지시문 템플릿을 만듭니다
prompt = ChatPromptTemplate.from_template(template)

# LLM. 구글 Gemini 모델을 설정합니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Retriever : 데이터베이스에서 정보를 찾아오는 '검색기' 역할을 부여합니다.
retriever = vectorstore.as_retriever()

# Combine Documents : 찾아온 여러 조각 문서들을 하나로 합치는 함수
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결. 랭체인의 파이프라인(체인)을 구성
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()} # 질문을 받으면 검색기(retriever)가 관련 정보를 찾고 합칩니다.
    | prompt # 질문 원문과 찾은 정보를 프롬프트에 넣습니다.
    | model # 모델(model)이 답변을 생성합니다.
    | StrOutputParser() # 결과를 문자열(StrOutputParser)로 변환합니다.
)

# Chain 실행. 최종 답변을 출력
rag_chain.invoke("격하 과정에 대해서 설명해주세요.")