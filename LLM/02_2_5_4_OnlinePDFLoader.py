"""
02_2_5_4_OnlinePDFLoader

"""
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문을 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
pages = loader.load() # 코렙에서 안됨.

len(pages)

pages[0].page_content[:1000]

"""
OnlinePDFLoader의 특징 요약
- 직접 연결: 파일을 내 컴퓨터에 다운로드할 필요 없이 URL 주소만으로 즉시 데이터를 가져옵니다.
- 최신성: arXiv 같은 논문 사이트나 정부 공고문 등 실시간으로 업데이트되는 온라인 문서를 읽어올 때 매우 유리합니다.
- 동작 원리: 웹에서 임시로 파일을 내려받은 후, Unstructured 엔진을 통해 텍스트와 구조를 분석합니다.
"""