"""
02_2_5_1_PyPDFLoader
PDF 파일을 읽어와서 AI가 처리하기 쉬운 형태의 문서 객체로 변환하는 가장 대중적인 방법입니다. 
PyPDFLoader를 사용하면 PDF의 페이지 단위로 데이터를 관리할 수 있습니다.
"""
# PDF 파일의 텍스트를 추출하기 위한 파이썬 라이브러리인 pypdf를 설치하는 명령어입니다.
# pip install pypdf
# pip install pymupdf

# LangChain에서 PDF 파일을 페이지별로 잘라서 불러오는 도구인 PyPDFLoader를 가져옵니다.
from langchain_community.document_loaders import PyPDFLoader # 용량이 큰 것은 읽지 못함.

pdf_filepath = './data/000660_SK_2023.pdf' # 불러올 PDF 파일의 경로를 변수에 저장. 25년자료는 못읽음

loader = PyPDFLoader(pdf_filepath) # 지정한 경로의 PDF를 읽을 수 있도록 로더를 초기화

# PDF 전체를 읽어서 페이지당 하나씩 Document 객체를 만들어 pages라는 리스트에 담습니다.
pages = loader.load()

# PDF 파일이 총 몇 페이지인지 확인합니다.
len(pages)

pages[10] # 전체 페이지 중 11번째 페이지(인덱스는 0부터 시작)의 내용을 출력
"""
- 페이지 단위 관리: PyPDFLoader의 가장 큰 장점은 문서가 몇 페이지에서 나온 
내용인지 메타데이터에 자동으로 기록해준다는 점입니다. 나중에 AI가 답변할 때 
"11페이지에 의하면~"이라고 답변의 근거를 밝히기 매우 좋습니다.
- 한글 지원: pypdf는 기본적으로 한글 텍스트 추출을 잘 지원하지만, 만약 PDF가 
글자가 아닌 '이미지(스캔본)'로 되어 있다면 텍스트가 추출되지 않을 수 있습니다.
그래서 PyMuPDFLoader사용한다.
"""