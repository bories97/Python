"""
# 텍스트 데이터를 숫자로 이루어진 벡터로 변환(Embedding)하고, 두 벡터 사이의 의미적 거리인 
# 코사인 유사도를 계산하는 전형적인 NLP(자연어 처리) 실습 과정
"""
# pip install langchain-openai

# OpenAI 모델을 사용하기 위한 LangChain 전용 패키지를 불러옵니다.
from langchain_openai import OpenAIEmbeddings

import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")

# OpenAI의 임베딩 모델 인스턴스를 생성합니다. 
# 기본적으로 'text-embedding-ada-002' 또는 'text-embedding-3-small' 모델이 사용됩니다.
embeddings_model = OpenAIEmbeddings()

# 5개의 한국어/영어 문장을 리스트 형태로 모델에 전달하여 숫자로 변환합니다.
# embed_documents는 여러 개의 문장을 한꺼번에 처리할 때 사용합니다.
embeddings = embeddings_model.embed_documents(
    [
    '안녕하세요!',
    '어! 오랜만이에요',
    '이름이 어떻게 되세요?',
    '날씨가 추워요',
    'Hello LLM!'
    ]
)

len(embeddings)  # 생성된 벡터의 개수(5개)를 확인합니다.
len(embeddings[0])  # 첫 번째 문장이 변환된 벡터의 차원 수(예: 1536차원)를 확인합니다.

print(embeddings[0][:20])   # 첫 번째 문장 벡터의 앞부분 20개 숫자만 샘플로 출력합니다.

# 검색하거나 비교하고 싶은 '질문' 하나를 벡터로 변환합니다.
# embed_query는 단일 문장을 임베딩할 때 최적화된 메서드입니다.
embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

embedded_query[:5]  # 질문 벡터의 앞부분 5개 숫자를 확인합니다.

# 코사인 유사도
import numpy as np
from numpy import dot   # 벡터의 내적을 계산합니다.
from numpy.linalg import norm  # 벡터의 크기(L2 Norm)를 계산합니다.

# 코사인 유사도 공식: 두 벡터의 내적을 각 벡터 크기의 곱으로 나눕니다.
def cos_sim(A, B):
    return dot(A, B) / (norm(A)*norm(B))

# 위에서 임베딩한 5개의 문장(embeddings)을 하나씩 꺼내어 질문(embedded_query)과 비교합니다.
for embedding in embeddings:
    # 유사도가 1에 가까울수록 의미가 비슷하고, 0에 가까울수록 관계가 없음을 뜻합니다.
    print(cos_sim(embedding, embedded_query))