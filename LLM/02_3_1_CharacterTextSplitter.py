"""
02_3_1_CharacterTextSplitter
LangChain을 활용하여 텍스트 파일을 불러오고, 이를 AI 모델이 처리하기 
좋은 크기로 쪼개는(Text Splitting) 전형적인 전처리 과정을 담고 있습니다.
"""
### 1. 데이터 로드 (TextLoader)
from langchain_community.document_loaders import TextLoader

# './data/history.txt' 파일을 UTF-8 인코딩으로 읽어옵니다.
# TextLoader: 로컬에 있는 .txt 파일을 읽어와서 LangChain의 Document 객체로 변환
loader = TextLoader('./data/history.txt',encoding="utf-8")
data = loader.load()

# 불러온 데이터의 첫 번째 문서(data[0])의 전체 글자 수를 출력합니다.
print(len(data[0].page_content))

# 데이터의 내용을 확인합니다.
data[0].page_content 

### 2. 글자 단위 분할 (CharacterTextSplitter - 무조건 분할)

# 각 문자를 구분하여 분할
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = '',           # 이 부분은 "글자 하나하나"를 기준으로 나누겠다는 뜻
    chunk_size = 500,         # 한 덩어리(Chunk)의 최대 길이는 500자마다 뚝뚝 끊어서 리스트로 만듭니다.
    chunk_overlap  = 100,     # 앞뒤 덩어리가 100자씩 겹치게 설정
    length_function = len,    # 길이를 재는 기준은 파이썬 기본 len 함수
)

texts = text_splitter.split_text(data[0].page_content)

len(texts)

len(texts[0])

texts[0]

### 3. 줄바꿈 단위 분할 (CharacterTextSplitter - 문장 보존)

# 줄바꿈 문자를 기준으로 분할
text_splitter = CharacterTextSplitter(
    separator = '\n',  # 글자 수를 채우더라도 가급적 엔터(\n)가 쳐진 곳에서 끊으라는 지시입니다.
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)

texts = text_splitter.split_text(data[0].page_content)

# 텍스트가 500자 단위로 쪼개져서 총 몇 개의 조각이 되었는지 보여줍니다.
len(texts) # 분할된 덩어리(Chunk)가 총 몇 개인지 확인

# len(texts[0]): 첫 번째 조각이 설정한 500자에 가깝게 잘 잘렸는지 확인하는 용도
len(texts[0]), len(texts[1]), len(texts[2]) # 각 덩어리의 글자 수를 확인

texts[0] # 첫 번째 덩어리의 실제 내용을 출력

"""
**"긴 역사 책 내용을 500자씩 100자 겹치면서 잘라내어, 
AI가 공부하기 좋은 요약 노트 조각으로 만드는 과정"**
"""