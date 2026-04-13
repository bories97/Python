"""
02_2_5_3_PyMuPDFLoader
PyMuPDFLoader는 이전에 사용했던 UnstructuredPDFLoader보다 속도가 매우 빠르고 메타데이터가 간결한 것이 특징
이것을 사용함.고해상도 이미지 처리가 필요할 때도 적합
"""
# pip install pymupdf
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = './data/000660_SK_2025.pdf'

loader = PyMuPDFLoader(pdf_filepath)

pages = loader.load()

len(pages)

pages[0].page_content

pages[0].metadata