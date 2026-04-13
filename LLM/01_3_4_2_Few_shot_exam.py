###  Few-shot 프롬프팅 기법은 고정된 예제
# 고정된 예제를 사용하는 방식과 질문에 맞춰 유사한 예제를 골라내는 동적 방식 두 가지를 모두 보여주고 있습니다

### 고정된 Few-shot 프롬프팅 (Static Few-shot)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 예제 정의. 모범 답안 리스트
examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
]

# 예제 프롬프트 템플릿 정의
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Few-shot 프롬프트 템플릿 생성. Few-shot 블록을 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 최종 프롬프트 템플릿 생성
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."), # 모델의 페르소나(교육자)를 설정
        few_shot_prompt,  # 앞에서 정의한 고정 예제들을 넣습니다.
        ("human", "{input}"),  # 사용자의 실제 질문({input})이 들어갈 자리를 만듭니다.
    ]
)

# 모델과 체인 생성
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma3:4b",  
    temperature=0,
    num_gpu=1,          # GPU 1개 사용 명시
)

# 프롬프트와 모델을 연결하여 하나의 실행 가능한 프로세스(체인)로 만듭니다.
chain = final_prompt | model

# 모델에 질문하기
result = chain.invoke({"input": "지구의 자전 주기는 얼마인가요?"})
print(result.content)

### 동적 Few-shot 프롬프팅 (Dynamic Few-shot)

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings

# 더 많은 예제 추가
examples = [
    {"input": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?", "output": "질소입니다."},
    {"input": "광합성에 필요한 주요 요소들은 무엇인가요?", "output": "빛, 이산화탄소, 물입니다."},
    {"input": "피타고라스 정리를 설명해주세요.", "output": "직각삼각형에서 빗변의 제곱은 다른 두 변의 제곱의 합과 같습니다."},
    {"input": "DNA의 기본 구조를 간단히 설명해주세요.", "output": "DNA는 이중 나선 구조를 가진 핵산입니다."},
    {"input": "원주율(π)의 정의는 무엇인가요?", "output": "원의 둘레와 지름의 비율입니다."},
]

# 벡터 저장소 생성. 예제 문장들을 하나로 합쳐서 컴퓨터가 이해할 수 있는 수학적 공간(벡터)에 넣을 준비를 합니다.
to_vectorize = [" ".join(example.values()) for example in examples]

# 한국어 문맥을 잘 이해하는 모델을 사용하여 텍스트를 숫자로 변환
embeddings = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# 변환된 숫자 데이터를 Chroma라는 가상의 지식 저장소(벡터 DB)에 저장
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# 예제 선택기 생성
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,           # 입력하신 그 데이터 리스트
    embeddings,         # 인텔 맥용 임베딩 모델
    Chroma,             # 벡터 저장소 클래스
    k=2                 # 선택할 예제 수
)
# 사용자가 질문을 던지면 embeddings 모델을 통해 질문의 의도를 파악하고,
# Chroma 저장소에서 가장 의미가 비슷한 예제를 k=2 (2개) 골라내는 검색기 역할을 합니다.

# Few-shot 프롬프트 템플릿 생성
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # examples 대신 **example_selector**를 넣었습니다. 이제 질문이 들어올 때마다 예제가 실시간으로 바뀝니다.
    example_selector=example_selector,
    # 고정 방식과 구조는 같지만, 내용물이 질문에 따라 동적으로 변합니다.
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# 최종 프롬프트 템플릿 생성
# 고정 방식과 구조는 같지만, 내용물이 질문에 따라 동적으로 변합니다.
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 과학과 수학에 대해 잘 아는 교육자입니다."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 모델과 체인 생성
chain = final_prompt | ChatOllama(model="gemma3:4b", temperature=0.0)

# 모델에 질문하기
# "태양계 행성"과 가장 비슷한 과학 예제(지구 대기, 광합성 등)를 먼저 검색합니다.
# 검색된 예제를 프롬프트에 끼워 넣습니다. 모델(Gemini)이 이를 참고해 최종 답변을 생성합니다.
result = chain.invoke({"input": "태양계에서 가장 큰 행성은 무엇인가요?"})
print(result.content)

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 
# pip uninstall torch torchvision torchaudio -y 버전 안바뀌면
import torch
print(torch.__version__)