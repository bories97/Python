"""
02_6_2_2_Multi_Query_ollama
앞의 예제를 올라마로 변경함. 지능은 떨어짐 속도는 컴 사양에 따라
"""

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyMuPDFLoader('./data/323410_카카오뱅크_2023.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)

# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore = FAISS.from_documents(documents,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE
)

# ollama run llama3.2 # 올라마 설치후 가벼운 모델. 터미널 재시작후 실행
# pip install -U langchain-ollama
# 멀티 쿼리 생성
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
import logging
import sys

# 1. 로깅 설정 (가장 먼저 실행)
# 기존 로깅 설정을 초기화하고 터미널(stdout)로 출력을 강제합니다.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
# MultiQueryRetriever 로거만 따로 한 번 더 설정
logger = logging.getLogger('langchain.retrievers.multi_query')
logger.setLevel(logging.INFO)

# ollama create gemma3:4b -f Modelfile  HuggingFace에서 사용 하기
llm = ChatOllama(
    model="gemma3:4b", 
    temperature=0
)

# MultiQueryRetriever 설정
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), 
    llm=llm,
)

# 5. 실행
question = '카카오뱅크의 최근 영업실적을 알려줘.'
unique_docs = retriever_from_llm.invoke(question) 
print(f"\n--- 검색된 문서 수: {len(unique_docs)} ---")

unique_docs[1]

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Prompt
# 2. 프롬프트 설정 (한국어 답변을 유도하도록 수정)
template = '''아래의 문맥(context)만을 바탕으로 질문에 답하세요:
{context}

질문: {question}

답변 (한국어로 자세히):'''

prompt = ChatPromptTemplate.from_template(template)

# Model
# 'llama3.2'는 사양이 낮은 컴퓨터에서도 잘 돌아가는 가벼운 모델입니다.
llm = ChatOllama(
    model="llama3.2", 
    temperature=0
)

def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

# Chain
chain = (
    {
        "context": retriever_from_llm | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Run
response = chain.invoke('카카오뱅크의 최근 영업실적을 요약해서 알려주세요.')
print(response)

