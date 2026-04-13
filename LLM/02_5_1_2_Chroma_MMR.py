"""
02_5_1_2_Chroma_MMR
PDF에서 데이터를 추출하는 실전 과정과, 직접 만든 문서를 벡터 저장소에 관리하는 
과정을 모두 담고 있습니다. 
특히 MMR 검색과 ID 부여를 통한 중복 방지가 핵심입니다.
"""
### 1. PDF 로드 및 텍스트 분할

# PyMuPDF를 사용해 PDF를 읽어오는 로더를 불러옵니다. (속도가 빠르고 정확함)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 카카오뱅크 ESG 보고서 PDF를 로드합니다.
loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')
data = loader.load()

# OpenAI 토큰 계산기 기준으로 텍스트를 쪼갭니다.
# 1,000토큰 크기로 자르되, 문맥을 위해 200토큰씩 겹치게 만듭니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

# 실제 문서를 쪼개어 'documents' 리스트에 담습니다.
documents = text_splitter.split_documents(data)

len(documents)

### 2. 임베딩 모델 및 벡터 저장소(Chroma) 생성

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 2. 임베딩 모델을 초기화합니다.
embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',     
    model_kwargs={'device':'cuda'},     
    encode_kwargs={'normalize_embeddings':True},
)

# 쪼개진 PDF 문서들을 Chroma DB에 저장합니다.
db2 = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name = 'esg',            # 저장소 내 그룹 이름
    persist_directory = './db/chromadb', # 하드디스크 저장 경로
    collection_metadata = {'hnsw:space': 'cosine'}, # 코사인 유사도 방식 사용
)

db2

### 3. 유사도 검색 vs MMR 검색

query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'

# [기본 검색] 질문과 가장 유사한 문서 4개를 가져옵니다.
docs = db2.similarity_search(query)

print(len(docs))
print(docs[0].page_content)
print(docs[-1].page_content)

# [MMR 검색] 질문과 유사하면서도, '서로 내용은 다른' 다양한 정보 위주로 4개를 뽑습니다.
# fetch_k=10: 후보를 10개 먼저 뽑은 뒤, 그중에서 가장 다양한 4개(k=4)를 선별합니다.
mmr_docs = db2.max_marginal_relevance_search(query, k=4, fetch_k=10)

print(len(mmr_docs))
print(mmr_docs[0].page_content)

print(mmr_docs[-1].page_content)
# MMR 검색 : 사용자에게 쿼리와 관련된 다양한 측면이나 정보를 제공하고자 할 때 유용합니다

### 4. 수동 문서 생성 및 ID 기반 저장 (중복 방지)
### Chroma 벡터 스토어에 문서와 메타데이터 저장

from langchain_core.documents import Document

# 코드에서 직접 Document 객체 리스트를 만듭니다. (메타데이터에 URL 포함)
documents = [
    Document(
        page_content="LangChain은 대규모 언어 모델(LLM)을 사용하는 애플리케이션을 개발하기 위한 프레임워크입니다.",
        metadata={
            "title": "LangChain 소개",
            "author": "AI 개발자",
            "url": "http://example.com/langchain-intro"
        }
    ),
    Document(
        page_content="벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색하는 데 특화된 데이터베이스 시스템입니다.",
        metadata={
            "title": "벡터 데이터베이스 개요",
            "author": "데이터 과학자",
            "url": "http://example.com/vector-db-overview"
        }
    ),
]

from langchain_community.vectorstores import Chroma

# Chroma 벡터 스토어에 문서와 메타데이터 저장. 위에서 만든 문서를 저장합니다.
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory="./chroma_db",  # 벡터 스토어를 디스크에 저장
    ids=["doc1", "doc2"]  # 중요: 고유 ID를 부여하여 재실행 시 중복 저장을 방지합니다!
)

### 5. 결과 출력 루프

query = "LangChain이란 무엇인가요?"

# 질문과 가장 비슷한 문서 2개를 찾아 'results' 변수에 담습니다
results = vectorstore.similarity_search(query, k=2)

# 검색 결과 리스트를 하나씩 꺼내어 정보를 출력합니다.
for doc in results:
    print(f"내용: {doc.page_content}")
    print(f"제목: {doc.metadata['title']}")
    print(f"저자: {doc.metadata['author']}")
    print(f"URL: {doc.metadata['url']}") # 이제 직접 넣었으므로 에러가 나지 않습니다.
    print("---")
    

"""
핵심 요약
1.MMR 검색: 정보가 편중되지 않게 "다양한 측면"의 답변 후보를 찾을 때 유리합니다.
2.ids 매개변수: ids=["doc1", "doc2"] 처럼 이름을 붙여주면, 같은 코드를 여러 번 
실행해도 DB에 똑같은 내용이 계속 쌓이는 것을 막아줍니다.
3.메타데이터 제어: PDF 로더가 주지 않는 정보(url 등)는 Document 객체를 생성할 때 
수동으로 넣어 관리할 수 있습니다.
"""