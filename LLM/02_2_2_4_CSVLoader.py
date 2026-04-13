"""
02_2_2_4_CSVLoader
CSV(표 형식) 파일을 불러와서 LangChain의 문서 객체로 변환하는 과정을 담고 있습니다. 
특히 CSV의 옵션을 바꿔가며 데이터가 어떻게 변하는지 실험하는 코드
"""
from langchain_community.document_loaders.csv_loader import CSVLoader

# 지정된 경로의 CSV 파일을 읽어올 준비를 합니다. 한국 공공기관 데이터는 보통 cp949(또는 euc-kr) 
# 인코딩이 많아 이렇게 설정해주면 한글이 깨지지 않습니다.
loader = CSVLoader(file_path='./data/한국주택금융공사_주택금융관련_지수_20211231.csv',
                   encoding='cp949')
# 파일을 실제로 읽어와 Document 객체 리스트로 저장
data = loader.load() 
len(data) # 전체 데이터 행의 개수를 확인

# 첫 번째 행의 데이터를 확인합니다. 
# 기본적으로 모든 열(Column)의 정보가 본문(page_content)에 들어갑니다.
data[0]

### 특정 열을 출처(Source)로 지정하기

# source_column='연도': 이 옵션을 추가하면, 생성된 문서의 메타데이터(metadata) 중 
# source 항목이 파일 경로가 아닌 해당 행의 '연도' 값으로 바뀝니다.
loader = CSVLoader(file_path='./data/한국주택금융공사_주택금융관련_지수_20211231.csv',
                   encoding='cp949', source_column='연도')
data = loader.load()

data[0] # 메타데이터에 source: 2004 (데이터에 따라 다름) 같은 식으로 찍히게 됩니다.

### 구분자(Delimiter) 변경 실험

# csv_args={'delimiter': '\n'}: CSV 데이터를 해석할 때 **줄바꿈(\n)**을 구분자로 사용하겠다는 뜻입니다.
loader = CSVLoader(file_path='./data/한국주택금융공사_주택금융관련_지수_20211231.csv',
                   encoding='cp949',
                   csv_args={
                       'delimiter': '\n',
                   })

data = loader.load()

# delimiter='\n' 설정이 적용되지 않았거나, 혹은 설정이 무시된 채 기본 쉼표(,) 구분자로 정상 파싱된 결과
data[0]