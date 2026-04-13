"""
02_3_2_RecursiveCharacterTextSplitter
**RecursiveCharacterTextSplitter**는 이전의 CharacterTextSplitter보다 훨씬 "똑똑한" 분할기입니다.
"""
### 1. 데이터 로드 (TextLoader)
from langchain_community.document_loaders import TextLoader

# './data/history.txt' 파일을 UTF-8 인코딩으로 읽어옵니다.
# TextLoader: 로컬에 있는 .txt 파일을 읽어와서 LangChain의 Document 객체로 변환
loader = TextLoader('./data/history.txt',encoding="utf-8")
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter: 이 분할기는 ["\n\n", "\n", " ", ""] 순서로 텍스트를 살펴봅니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, # 하나의 덩어리(Chunk)를 최대 500자 정도로 맞춥니다.
    # 덩어리 사이의 연결 고리를 위해 앞 덩어리의 끝 100자를 다음 덩어리 시작에 포함시킵니다.
    chunk_overlap  = 100, 
    length_function = len,
)

# 설정한 규칙대로 전체 텍스트를 쪼개어 리스트 형태로 texts에 저장합니다.
texts = text_splitter.split_text(data[0].page_content)

# 총 몇 개의 조각으로 나뉘었는지 확인합니다.
len(texts)

# len(texts[0]): 첫 번째 조각의 길이를 확인
len(texts[0]), len(texts[1]), len(texts[2])

texts[0] # 문서의 가장 앞부분

"""
1.문서의 시작 부분: 제목이나 도입부가 포함됩니다.
2.문맥 보존: 500자에 도달했을 때 무조건 자르는 것이 아니라, 가장 가까운 줄바꿈이나 
띄어쓰기 지점을 찾아 자르기 때문에 문장이 비교적 깔끔하게 마무리된 상태로 들어있을 것입니다.
3.내용적 특징: history.txt가 역사 데이터라면, 특정 시대의 배경이나 정의에 대한 
설명이 500자 내외로 담기게 됩니다.
"""

"""
CharacterTextSplitter와 다른 점은 무엇인가요?
이전 코드(CharacterTextSplitter)는 지정한 구분자(예: \n)가 없으면 설정값을 무시하고 
넘어가거나 무조건 잘라버릴 수 있지만, 지금 사용하신 Recursive 방식은 여러 구분자를 
순차적으로 적용하기 때문에 **"의미 단위"**를 훨씬 더 잘 보존합니다. AI에게 학습시키거나 
질문 답변(RAG) 시스템을 만들 때 훨씬 유리한 방식입니다.
"""