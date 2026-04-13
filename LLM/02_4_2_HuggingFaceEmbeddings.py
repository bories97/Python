"""
02_4_2_HuggingFaceEmbeddings
**내 컴퓨터의 자원(CPU)을 사용해서 한국어 전용 오픈소스 모델(HuggingFace)**로 
임베딩을 생성하는 방법
"""
### 1. 허깅페이스 임베딩 모델 설정
# 1. HuggingFace의 오픈소스 모델을 사용하기 위한 클래스를 가져옵니다.
from langchain_community.embeddings import HuggingFaceEmbeddings

# 2. 임베딩 모델을 초기화합니다.
embeddings_model = HuggingFaceEmbeddings(
    # 'jhgan/ko-sroberta-nli': 한국어 문장 임베딩에 최적화된 성능을 내는 모델 경로입니다.
    model_name='BAAI/bge-m3',     
    # 'device':'cuda': NVIDIA 그래픽카드를 사용
    model_kwargs={'device':'cuda'},     
    # 'normalize_embeddings':True: 생성된 벡터의 길이를 1로 맞춥니다(정규화). 
    # 이렇게 하면 코사인 유사도 계산이 더 정확해지고 단순해집니다.
    encode_kwargs={'normalize_embeddings':True},
)
# 3. 모델 설정 상태를 확인합니다.
embeddings_model

### 2. 문서 및 질문 임베딩 생성

# 4. 준비된 5개의 한국어 문장을 숫자의 리스트(벡터)로 변환합니다.
embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

# 5. 생성된 결과의 크기를 확인합니다. 
# (5개의 문장, 각 문장의 벡터 차원 수)를 출력합니다.
# 이 모델의 경우 각 문장은 768개의 숫자로 변환됩니다.
len(embeddings) 
len(embeddings[0])

# 7. 수치 계산을 위한 numpy 라이브러리에서 행렬 곱(dot)과 벡터 크기(norm) 함수를 가져옵니다.
import numpy as np
from numpy import dot
from numpy.linalg import norm

# 8. 코사인 유사도 함수 정의: 1에 가까울수록 두 문장의 의미가 비슷함을 의미합니다.
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

# 6. 검색할 질문인 '첫인사를 하고 이름을 물어봤나요?'를 벡터로 변환합니다.
embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')

# 9. 반복문을 통해 처음에 임베딩했던 5개의 문장과 질문 사이의 유사도를 하나씩 출력합니다.
for embedding in embeddings:
    # 질문과 각 문장 사이의 의미적 거리(유사도 점수)가 출력됩니다.
    print(cos_sim(embedding, embedded_query))

"""
- 한국어 특화: 사용된 BAAI/bge-m3 모델은 한국어의 미묘한 차이를 잘 이해합니다. 
예를 들어 '안녕하세요'와 '첫인사'라는 단어가 달라도 의미적으로 가깝다는 것을 점수로 
보여줍니다.
- 로컬 실행: 인터넷 연결이나 API 키 없이도 내 컴퓨터 안에서 모든 계산이 이루어지므로 
보안에 유리하고 무료입니다.
- 정규화(normalize_embeddings): 이미 모델 설정에서 벡터 길이를 1로 맞췄기 때문에, 
사실 코사인 유사도는 단순한 내적(dot) 계산만으로도 유사한 결과를 얻을 수 있을 만큼 
최적화되어 있습니다.
"""