"""
02_5_2_Chroma_FAISS
Meta(Facebook)에서 개발한 고성능 벡터 검색 라이브러리인 FAISS를 사용하여 
RAG(검색 증강 생성) 시스템을 구축하는 과정을 담고 있습니다. 앞서 사용했던 
Chroma와는 또 다른 강력한 검색 엔진인 FAISS의 특징이 잘 드러나 있습니다.
"""
### 1. PDF 로드 및 문서 분할

# 1. 라이브러리 임포트
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 2. 카카오뱅크 보고서 PDF를 불러옵니다.
loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')
data = loader.load()

# 3. 텍스트를 1,000 토큰 단위로 자르되, 맥락을 유지하기 위해 200 토큰씩 겹치게 나눕니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

# 4. 쪼개진 문서 조각들을 생성합니다.
documents = text_splitter.split_documents(data)
len(documents)

### 2. FAISS 벡터스토어 생성 및 임베딩
# pip install faiss-gpu 리눅스 계열
# conda install -c pytorch faiss-gpu 아나콘다(Conda) 사용자라면 (GPU 지원 가능)
# pip install faiss-cpu 

# 벡터스토어 db 인스턴스를 생성
# 5. FAISS 검색 엔진과 거리 측정 전략, 임베딩 모델을 가져옵니다.
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

# 6. 한국어 문장 임베딩 모델을 설정합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# 7. FAISS 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(
    documents,                         # 쪼개진 문서들
    embedding = embeddings_model,      # 임베딩 모델
    distance_strategy = DistanceStrategy.COSINE # 유사도 계산 방식을 '코사인 유사도'로 설정
)

vectorstore

# 8. 현재 설정된 거리 측정 전략(COSINE)을 확인합니다.
vectorstore.distance_strategy

### 3. 유사도 및 MMR 검색

# 9. 질문을 정의합니다.
query = '카카오뱅크가 중대성 평가를 통해 도출한 7가지 중대 주제는 무엇인가?'

# 10. 일반 유사도 검색: 질문과 가장 비슷한 문서 조각들을 가져옵니다.
docs = vectorstore.similarity_search(query)

print(len(docs))
print(docs[0].page_content)

# 11. MMR 검색: 질문과 유사하면서도 서로 중복되지 않는 '다양한' 문서 4개를 추출합니다.
# fetch_k=10: 후보 10개를 먼저 찾은 후, 그 안에서 가장 다양한 4개를 골라냅니다.
mmr_docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)

print(len(mmr_docs))
print(mmr_docs[0].page_content)

### 4. 벡터스토어 저장 및 불러오기

# 12. 생성된 FAISS 인덱스를 로컬 컴퓨터의 './db/faiss' 폴더에 저장합니다.
# (나중에 코드를 다시 실행할 때 임베딩 과정을 생략할 수 있어 시간을 절약합니다.)
vectorstore.save_local('./db/faiss')

# 13. 저장된 FAISS 인덱스를 다시 불러옵니다.
# allow_dangerous_deserialization=True: 신뢰할 수 있는 로컬 파일이므로 역직렬화를 허용합니다.
db3 = FAISS.load_local('./db/faiss', embeddings_model, allow_dangerous_deserialization=True)

# 14. 불러온 DB 객체를 확인합니다.
db3

"""
- FAISS의 강점: 대규모 데이터셋에서 검색 속도가 매우 빠릅니다. Chroma가 사용하기 
편리하다면, FAISS는 대량의 문서를 다룰 때 성능상 이점이 큽니다.
- DistanceStrategy.COSINE: 검색의 정확도를 높이기 위해 문장 사이의 '각도'를 비교
하는 코사인 유사도를 명시적으로 사용했습니다.
- 로컬 저장/로드: save_local과 load_local을 통해 비싼 연산인 임베딩 과정을 매번 
반복하지 않고, 이미 수치화된 데이터를 재사용할 수 있게 했습니다.
"""

### 확인
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'
# [기본 검색] 질문과 가장 유사한 문서 4개를 가져옵니다.
docs = db3.similarity_search(query)
print(len(docs))
print(docs[0].page_content)

print(docs[-1].page_content)