"""
02_5_1_1_Chroma_Similarity
텍스트 데이터를 불러와 잘게 나누고, 이를 오픈소스 임베딩 모델을 통해 
**Chroma라는 벡터 데이터베이스(Vector DB)**에 저장한 뒤 질문과 가장 
유사한 답변을 찾아내는 **RAG(검색 증강 생성)**의 핵심 과정을 담고 
있습니다.
"""
### 1. 데이터 로드 및 텍스트 분할
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
# 1. 텍스트 파일 로더와 텍스트 분할기, 제미나이 채팅 모델, 크로마 DB 라이브러리를 가져옵니다.
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# 2. './data/history.txt' 파일을 UTF-8 인코딩으로 읽어옵니다.
loader = TextLoader('./data/history.txt',encoding="utf-8")
data = loader.load()

# 3. 텍스트를 토큰(Token) 기준으로 효율적으로 나누기 위한 설정입니다.
# 약 250 토큰씩, 이전 조각과 50 토큰이 겹치도록 설정하여 문맥 유실을 방지합니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

# 4. 불러온 파일의 전체 텍스트 내용을 설정한 규칙에 따라 여러 개의 조각(texts)으로 나눕니다.
texts = text_splitter.split_text(data[0].page_content)

# 나뉜 첫 번째 텍스트 조각을 확인합니다.
texts[0]

### 2. 한국어 임베딩 모델 설정

# 5. 오픈소스인 허깅페이스 임베딩 라이브러리를 가져옵니다.
from langchain_community.embeddings import HuggingFaceEmbeddings

# 2. 임베딩 모델을 초기화합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',     
    model_kwargs={'device':'cuda'},     
    encode_kwargs={'normalize_embeddings':True},
)

### 3. Chroma 벡터 데이터베이스 구축

# 7. 텍스트 조각들을 벡터 DB인 Chroma에 저장합니다.
db = Chroma.from_texts(
    texts,                   # 쪼개진 텍스트 조각들
    embeddings_model,        # 텍스트를 숫자로 바꿀 모델
    collection_name='history', # DB 안에서의 저장소(테이블) 이름
    persist_directory='./db/chromadb', # DB를 하드디스크에 저장할 경로 (나중에도 사용 가능)
    collection_metadata={'hnsw:space': 'cosine'}, # 유사도 계산 방식을 '코사인 유사도'로 설정
)

# 8. 생성된 DB 객체 정보를 확인합니다.
db

# 9. AI에게 물어볼 질문을 정의합니다.
query = '누가 한글을 창제했나요?'

# 10. 질문(query)을 임베딩 모델로 수치화한 뒤, DB에 저장된 조각들 중 
# 의미적으로 가장 유사한 상위 조각들을 검색해 가져옵니다.
docs = db.similarity_search(query)

# 11. 가장 유사도가 높은 첫 번째 조각(docs[0])의 실제 텍스트 내용을 출력합니다.
print(docs[0].page_content)

"""
코드의 핵심 포인트
- 벡터 저장소 (Chroma): 텍스트를 단순 저장하는 것이 아니라, 임베딩 모델을 
통해 **의미가 담긴 숫자(벡터)**로 저장합니다.
- 코사인 유사도 (hnsw:space: cosine): 질문과 답변이 얼마나 같은 방향성(의미)
을 가졌는지 측정하는 최적의 방식입니다.
- RAG의 시작: 이제 docs[0]에 담긴 정보를 제미나이(ChatGoogleGenerativeAI)에게 
전달하면, AI가 이 내용을 바탕으로 정확한 답변을 생성하게 됩니다.
"""