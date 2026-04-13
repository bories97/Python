### Few-shot Prompt
# LangChain의 핵심 기능 중 하나인 Few-shot 프롬프팅과 **의미론적 유사성 예제 
# 선택기(Semantic Similarity Example Selector)**를 활용하는 아주 좋은 예시

from langchain_core.prompts import PromptTemplate

# 개별 예제들이 화면에 출력될 형식을 지정
example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")

# AI에게 학습시키거나 참고시킬 '모범 사례' 리스트입니다. 
# 지구 과학, 수학, 생물 등 다양한 분야의 질문과 답변 쌍을 담고 있습니다.
examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

from langchain_core.prompts import FewShotPromptTemplate

# FewShotPromptTemplate을 생성합니다.
# 고정된 Few-shot 프롬프트 생성
prompt = FewShotPromptTemplate(
    examples=examples,              # 위에서 만든 예제 리스트를 사용합니다.
    example_prompt=example_prompt,  # 각 예제를 어떤 모양으로 보여줄지 결정합니다.
    suffix="질문: {input}",          # 예제들이 쭉 나온 뒤 마지막에 사용자의 실제 질문을 붙입니다.
    input_variables=["input"],      # 사용자가 입력할 변수 이름은 'input'입니다.
)
# 모든 예제를 프롬프트에 한꺼번에 집어넣는 방식입니다. 
# 질문이 적을 때는 유용하지만, 예제가 많아지면 토큰(비용)이 많이 듭니다.

# 새로운 질문에 대한 프롬프트를 생성하고 출력합니다.
print(prompt.invoke({"input": "화성의 표면이 붉은 이유는 무엇인가요?"}).to_string())

# pip install langchain-community
# pip install sentence-transformers
# pip install chromadb

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings

# 한국어 성능이 좋은 ko-sroberta-multitask 모델을 사용
embeddings_model = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')

# SemanticSimilarityExampleSelector를 초기화합니다.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,            # 전체 예제 리스트 중에서
    embeddings_model,    # 문장 의미를 분석할 모델을 사용하고
    Chroma,              # 분석된 데이터를 저장할 데이터베이스(Chroma-벡터 저장소)에 넣은 뒤
    k=1,                 # 사용자의 질문과 가장 의미가 비슷한 예제 딱 '1개'만 골라라!
)
# 이 부분이 똑똑한 점입니다. 수천 개의 예제가 있어도, 
# 사용자의 질문과 가장 관련 있는 것만 골라서 AI에게 보여주게 됩니다.

# 새로운 질문에 대해 가장 유사한 예제를 선택합니다.
question = "화성의 표면이 붉은 이유는 무엇인가요?"
selected_examples = example_selector.select_examples({"question": question})

# 선택된 예제의 키(question, answer)와 내용(v)을 화면에 출력
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")

"""
**"모든 예제를 무식하게 다 보여주는 게 아니라, 사용자의 질문과 가장 관련 
있는 '모범 답안' 하나만 쏙 골라서 AI에게 참고하라고 전달하는 방식"
"""