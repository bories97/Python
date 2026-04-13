"""
RAG(Retrieval-Augmented Generation, 검색 증강 생성) 시스템의 핵심 과정을 담고 있습니다. 
특히 검색된 문서에서 불필요한 내용을 쳐내고 핵심만 남기는 문맥 압축(Contextual Compression) 기술이 
적용된 코드입니다.
"""
# 1. PDF 파일 로드: PyMuPDF를 사용하여 지정된 경로의 PDF 내용을 불러옵니다.
loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')
data = loader.load()

# 2. 텍스트 분할기 설정: 긴 문서를 처리하기 쉬운 조각(Chunk)으로 나눕니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,       # 한 조각당 최대 1,000 토큰
    chunk_overlap=200,      # 조각 간의 문맥 연결을 위해 200 토큰씩 겹치게 설정
    encoding_name='cl100k_base' # OpenAI 모델에서 사용하는 인코딩 방식 기준
)

# 3. 실제 분할 실행: 로드된 데이터를 설정한 기준에 따라 리스트 형태의 문서들로 나눕니다.
documents = text_splitter.split_documents(data)

## 임베딩 및 벡터 저장소 구축 (Vector Store)

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 성능이 좋은 BAAI/bge-m3 모델을 사용하여 문장을 벡터로 변환합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

# 5. 벡터 저장소 생성: 분할된 문서들을 임베딩하여 FAISS(메모리 기반 벡터 DB)에 저장합니다.
# 거리 측정 방식은 '코사인 유사도(COSINE)'를 사용합니다.
vectorstore = FAISS.from_documents(documents,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE
)

## 언어 모델(LLM) 및 기본 리트리버 설정

from langchain_google_genai import ChatGoogleGenerativeAI
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

# 6. Gemini 모델 설정: Google의 Gemini 2.5 Flash 모델을 사용하여 답변 생성 및 문서 압축을 수행합니다.
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', 
    temperature=0,            # 창의성 0 (가장 사실적이고 일관된 답변 유도)
    max_tokens=500,
)

# 7. 기본 검색기(Retriever) 설정: 질문과 관련된 문서를 DB에서 찾아오는 역할을 합니다.
# MMR(Maximal Marginal Relevance) 방식은 유사도뿐만 아니라 '다양성'을 고려해 중복된 정보를 피합니다.
base_retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k':7, 'fetch_k': 20} # 20개를 후보로 뽑고 그중 가장 관련성 높은 7개를 최종 반환
)

# 8. 질문 실행: 질문에 대해 관련된 문서 7개를 찾아옵니다.
question = '카카오뱅크의 최근 영업실적을 알려줘.'
docs = base_retriever.invoke(question)
print(len(docs)) # 출력: 7

# 문서 압축기를 연결하여 구성

# langchain-classic 패키지 사용
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# 9.검색해오는 문서의 수를 2개로 명시적으로 제한합니다.
# LLM이 처리해야 할 데이터양을 미리 줄여두는 효과적인 조치입니다.
base_retriever.search_kwargs = {"k": 2}

# 10. 문서 압축기 생성: LLM을 이용해 문서 내에서 질문('영업실적')과 관련된 내용만 
# 추출하는 엔진을 만듭니다.
compressor = LLMChainExtractor.from_llm(llm)

# 11. 압축 리트리버 결합: 기본 검색기와 압축기를 하나로 묶습니다.
# 작동 순서: [DB 검색(2개)] -> [LLM이 각 문서에서 핵심만 추출] -> [압축된 문서 반환]
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# 12. 압축 검색 실행: 질문을 던져 최종적으로 정제된(압축된) 문서들을 받습니다.
compressed_docs = compression_retriever.invoke(question)

# 13. 결과 확인
print(len(compressed_docs)) # 최대 2개의 압축된 문서 객체가 담깁니다.

compressed_docs