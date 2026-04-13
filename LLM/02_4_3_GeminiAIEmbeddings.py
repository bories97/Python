"""
02_4_3_GeminiAIEmbeddings
제미나이 임베딩 모델을 활용해 **텍스트를 숫자로 변환(Embedding)**하고, 
그 결과값들 사이의 유사도(Similarity)를 계산하는 예제입니다.
"""
### 1. 모델 설정 및 데이터 변환

# 1. 구글 제미나이 임베딩 기능을 사용하기 위해 필요한 도구를 가져옵니다.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# pip install google-generativeai

# 2. 제미나이 임베딩 모델을 초기화합니다.
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# 3. 여러 개의 문장을 한꺼번에 벡터(숫자 리스트)로 변환합니다.
embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

# 4. 결과 확인: (생성된 벡터의 개수, 각 벡터의 차원 수)를 출력합니다.
# 결과는 (5, 3072)이 나올 것입니다. (5개 문장, 각 3072개의 숫자)
len(embeddings)
len(embeddings[0])

# 5. 첫 번째 문장('안녕하세요!')이 변환된 숫자들 중 앞의 20개만 샘플로 출력해 봅니다.
print(embeddings[0][:20])

### 2. 질문 임베딩 (Query Embedding)

# 6. 사용자의 질문을 벡터로 변환합니다. 
# 문서 뭉치가 아닌 '질문 하나'를 변환할 때는 embed_query 메서드를 사용합니다.
embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

# 7. 변환된 질문 벡터의 앞부분 숫자 5개만 확인합니다.
embedded_query[:5]

### 3. 코사인 유사도 계산 (Cosine Similarity)
# 두 벡터가 얼마나 비슷한 방향을 가리키고 있는지를 수학적으로 계산하는 과정입니다.

# 8. 수치 계산을 위한 라이브러리인 numpy를 가져옵니다.
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))