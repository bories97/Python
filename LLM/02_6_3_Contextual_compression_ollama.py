"""
02_6_3_Contextual_compression의 Docstring
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

embeddings_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

vectorstore = FAISS.from_documents(documents,
    embedding = embeddings_model,
    distance_strategy = DistanceStrategy.COSINE
)

from langchain_ollama import ChatOllama

question = '카카오뱅크의 최근 영업실적을 알려줘.'

# ollama create bllossom-custom -f Modelfile  HuggingFace에서 사용 하기
llm = ChatOllama(
    model="bllossom-custom", 
    temperature=0
)

base_retriever = vectorstore.as_retriever(
                                search_type='mmr',
                                search_kwargs={'k':7, 'fetch_k': 20})

# v1.0: invoke() 메서드 사용
docs = base_retriever.invoke(question)
print(len(docs))

# 문서 압축기를 연결하여 구성

# v1.0: langchain-classic 패키지 사용
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# v1.0: invoke() 메서드 사용
compressed_docs = compression_retriever.invoke(question)
print(len(compressed_docs))

compressed_docs