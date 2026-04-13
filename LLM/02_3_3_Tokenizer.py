"""
02_3_3_Tokenizer
OpenAI의 모델(제미나이 등)이 인식하는 단위인 '토큰(Token)'을 기준으로 텍스트를 
분할하는 방식입니다. LLM 비용 관리와 성능 최적화에 아주 효율적인 코드입니다.
"""
### 1. 데이터 로드 (TextLoader)
from langchain_community.document_loaders import TextLoader

loader = TextLoader('./data/history.txt',encoding="utf-8")
data = loader.load()

from langchain_text_splitters import CharacterTextSplitter

# .from_tiktoken_encoder: OpenAI의 토큰화 도구인 'tiktoken'을 사용하여 분할기를 만듭니다.
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,            # 한 덩어리(Chunk)를 약 600 토큰 내외로 맞춥니다.
    chunk_overlap=200,         # 덩어리 간의 문맥 연결을 위해 200 토큰씩 겹치게 합니다.
    encoding_name='cl100k_base' # GPT-4, gpt-3.5-turbo에서 사용하는 최신 인코딩 방식입니다.
)
"""
왜 토큰 기준인가요? AI 모델은 "사과"라는 글자보다 이를 수치화한 "토큰" 단위를 사용합니다. 
한글은 글자 수와 토큰 수가 다르기 때문에, tiktoken을 사용하면 모델의 입력 한계를 넘지 
않도록 훨씬 정밀하게 조절할 수 있습니다.
"""
# split_documents: 앞서 로드한 data(Document 객체)를 규칙에 따라 쪼갭니다.
docs = text_splitter.split_documents(data)
# 총 몇 개의 문서 조각(docs)이 만들어졌는지 개수를 확인합니다.
len(docs)

# 첫 번째 조각(docs[0])의 실제 '글자 수'를 출력합니다.
print(len(docs[0].page_content))
# 첫 번째 조각의 내용과 메타데이터(파일 경로 등)를 확인합니다.
docs[0]

# 두 번째 조각(docs[1])의 글자 수를 출력합니다.
print(len(docs[1].page_content))
# 두 번째 조각의 내용을 확인합니다. (이때 docs[0]의 뒷부분 200토큰 정도가 포함되어 있습니다.)
docs[1]
"""
핵심 요약
1.정밀도: 단순히 글자 수로 자르면 한글 단어가 깨질 수 있지만, tiktoken은 AI 모델이 
이해하는 단위로 자르기 때문에 답변 품질이 좋아집니다.
2.연속성: chunk_overlap=200 덕분에 docs[0]에서 끊긴 이야기가 docs[1]의 앞부분에서 
다시 이어져, AI가 문맥을 잃어버리지 않습니다.
3.객체 유지: split_text가 아닌 split_documents를 사용했기 때문에, 원본 파일의 
이름이나 위치 정보(metadata)가 각 조각마다 그대로 따라다닙니다.
"""