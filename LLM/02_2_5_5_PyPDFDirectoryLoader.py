"""
02_2_5_5_PyPDFDirectoryLoader
PyPDFDirectoryLoader는 지정된 디렉토리(폴더) 안에 있는 모든 
PDF 파일을 한꺼번에 읽어오는 매우 편리한 도구입니다.
"""
from langchain_community.document_loaders import PyPDFDirectoryLoader

# 한 줄로 폴더 내 수십 개의 PDF를 한 번에 리스트로 만듭니다.
loader = PyPDFDirectoryLoader('./data/')
data = loader.load()

len(data)

data[0]

# 로드된 전체 문서 리스트 중에서 가장 마지막에 위치한 문서 객체를 의미
data[-1]

# 마지막 데이터의 출처
print(f"출처 파일: {data[-1].metadata['source']}")
print(f"페이지 번호: {data[-1].metadata['page'] + 1} 페이지") # 0부터 시작하므로 +1