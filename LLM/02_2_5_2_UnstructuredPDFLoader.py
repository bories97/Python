"""
02_2_5_2_UnstructuredPDFLoader
단순히 텍스트만 긁어오는 게 아니라, 해당 데이터의 **출처와 속성(메타데이터)**을 함께 저장합니다.
"""
# pip install "unstructured[pdf]"  

from langchain_community.document_loaders import UnstructuredPDFLoader

pdf_filepath = './data/000660_SK_2023.pdf'

# 전체 텍스트를 단일 문서 객체로 변환
loader = UnstructuredPDFLoader(
    pdf_filepath,
    strategy="fast",      # 'hi_res' 대신 'fast'를 쓰면 복잡한 그래픽 분석을 건너뜁니다.
)
pages = loader.load()

len(pages)

pages[0].page_content[:1000]

pages[0]

pdf_filepath = './data/000660_SK_2023.pdf'

# 텍스트 조각(chunk)를 별도 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_filepath, mode='elements')
pages = loader.load()

len(pages)

pages[100]

# 101번째 조각의 순수 텍스트만 확인
print(pages[100].page_content)

# 101번째 조각이 몇 페이지에 있는지 확인
print(pages[100].metadata['page_number'])

# 101번째 조각의 성격(제목인지 본문인지) 확인
print(pages[100].metadata['category'])