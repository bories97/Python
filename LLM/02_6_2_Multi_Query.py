"""
02_6_2_Multi_Query
Multi-Query Retrieval(다중 쿼리 검색) 방식의 RAG(Retrieval-Augmented Generation) 
시스템을 구축하는 과정
"""
### 1. 문서 로드 및 텍스트 분할

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 지정된 경로의 PDF 파일을 읽어오는 로더를 생성
loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')
data = loader.load() # PDF의 내용을 불러와 메모리에 저장
# 텍스트를 자르는 도구
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, # 한 번에 최대 1000토큰씩 자릅니다.
    chunk_overlap=200, # 문맥 연결을 위해 앞뒤 청크 간에 200토큰씩 겹치게 합니다.
    encoding_name='cl100k_base'
)

# 불러온 문서를 설정한 규칙에 따라 여러 개의 작은 조각(documents)으로 나눕니다.
documents = text_splitter.split_documents(data)

### 2. 임베딩 및 벡터 스토어 구축

# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 성능이 좋은 AAI/bge-m3 모델을 사용하여 문장을 벡터로 변환합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# 오픈소스 벡터 저장소인 FAISS를 생성합니다.
vectorstore = FAISS.from_documents(documents,
    embedding = embeddings_model,
    # 문장 간의 유사도를 측정할 때 '코사인 유사도' 방식을 사용하도록 설정합니다.
    distance_strategy = DistanceStrategy.COSINE
)

### 3. Multi-Query Retriever 설정

# 멀티 쿼리 생성
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# 1. 로깅 설정 (가장 먼저 실행)
logging.basicConfig() # 노트북에서

# 멀티 쿼리가 어떻게 생성되는지 로그를 통해 확인하기 위한 설정입니다.
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# 최신 Gemini 모델을 호출
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
)

# MultiQueryRetriever 설정
# 질문을 확장할 LLM(Gemini)과 문서가 저장된 리트리버(FAISS)를 연결
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), 
    llm=llm,
)

### 4. 검색 실행 (중간 단계)

question = '카카오뱅크의 최근 영업실적을 알려줘.'

# Gemini가 질문을 여러 개로 만든 후, 각각에 대해 관련 문서를 찾아 중복을 제거하고 가져옵니다.
unique_docs = retriever_from_llm.invoke(question) 
print(f"\n--- 검색된 문서 수: {len(unique_docs)} ---")

### 5. RAG 체인 구성 및 최종 실행

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# AI에게 줄 명령문(Prompt)입니다. "준비된 문맥(context)만 참고해서 질문에 답하라"고 제약을 줍니다.
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
prompt = ChatPromptTemplate.from_template(template)

# Model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0,
)

# 검색된 여러 문서 조각들을 하나의 긴 텍스트로 합쳐주는 함수
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

# Chain 구성 (| 기호는 단계를 연결합니다):
chain = (
    # 문서를 찾아서 텍스트로 변환합니다, 'question': RunnablePassthrough(): 사용자의 질문을 그대로 전달
    {'context': vectorstore.as_retriever() | format_docs, 'question': RunnablePassthrough()}
    # 프롬프트에 값을 넣고 -> Gemini가 읽고 -> 결과를 문자열로 출력합니다.
    | prompt
    | llm
    | StrOutputParser()
)

# Run
response = chain.invoke('카카오뱅크의 최근 영업실적을 요약해서 알려주세요.')
print(response)