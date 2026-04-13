"""
02_6_1_Vector_Store
PDF 문서에서 정보를 찾아 답변하는 RAG(Retrieval-Augmented Generation) 
시스템의 전 과정을 담고 있습니다.
"""
### 1. 데이터 로드 및 텍스트 분할 (Data Ingestion)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 지정된 경로의 PDF 파일을 불러옵니다. PyMuPDF는 속도가 빠르고 텍스트 추출 성능이 좋은 라이브러리입니다.
loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')

data = loader.load()

# RecursiveCharacterTextSplitter: 긴 문서를 일정한 길로 자릅니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200, 
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)
len(documents)

### 2. 임베딩 및 벡터스토어 저장 (Embedding & Vector Store)

# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# Facebook에서 만든 고속 유사도 검색 라이브러리인 FAISS에 문서들을 저장합니다.
vectorstore = FAISS.from_documents(documents,
        embedding = embeddings_model,
        distance_strategy = DistanceStrategy.COSINE
)

# 검색 쿼리
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

### 3. 다양한 검색(Retrieval) 전략 테스트

# 가장 유사도가 높은 문장을 하나만 추출. k=1을 주어 가장 비슷한 문서 1개만 가져옵니다.
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

docs = retriever.invoke(query)
print(len(docs))
docs[0]

# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.invoke(query)
print(len(docs))
docs[-1]

# Similarity score threshold (유사도 점수 이상인 문서만을 대상으로 추출)
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.3}  # 임계값은 0.3으로 설정
)

docs = retriever.invoke(query)
print(len(docs))
docs[0]


# 문서 객체의 metadata를 이용한 필터링
retriever = vectorstore.as_retriever(
    # Filter: PDF 버전 정보 같은 메타데이터를 기준으로 특정 조건의 문서만 골라냅니다.
    search_kwargs={'filter': {'format':'PDF 1.5'}}
)

docs = retriever.invoke(query)
print(len(docs))
docs[0]

### 4. RAG 체인 구성 및 실행 (Generation)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Retrieval(검색)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.5} # 상위 5개의 관련성이 높으면서도 다양한 문서를 선택
)

docs = retriever.invoke(query)

# Prompt
# AI에게 줄 명령서(프롬프트) 양식을 만듭니다. {context}에는 검색된 문서 내용이, 
# {question}에는 사용자의 질문이 들어갑니다.
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# 구글의 Gemini 모델을 연결
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0, # AI의 창의성을 0으로 만들어, 항상 일관되고 사실적인 답변을 하도록 설정
    max_tokens=2000,
)

# 검색된 여러 개의 문서 덩어리를 하나로 합쳐서 AI에게 전달하기 좋게 문자열로 만듭니다.
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({'context': (format_docs(docs)), 'question':query})
response
