"""
02_2_3_DirectoryLoader
특정 폴더 안에 있는 여러 텍스트 파일을 한 번에 불러오기 위한 설정입니다.
"""
# DirectoryLoader는 기본적으로 다양한 파일 형식을 다루기 위해 unstructured 라이브러리를 
# 사용하곤 합니다. (다만, 아래 코드처럼 TextLoader를 직접 지정하면 필수 사항은 아니지만 
# 설치해두면 유용합니다.)
# 

# 파일 경로를 다루고(os), 특정 패턴(예: *.txt)에 맞는 파일 목록을 뽑아내기 
# 위해 사용하는 파이썬 기본 도구들입니다.
import os
from glob import glob
# PC에서는 현재 작업 디렉토리 ./
files = glob(os.path.join('./data/', '*.txt'))
files # 화면에 출력

### 로더(Loader) 설정
# 폴더를 읽는 도구(DirectoryLoader)와 개별 텍스트 파일을 읽는 도구(TextLoader)를 각각 가져옵니다.
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader(
    path='./data/', # 읽어올 파일들이 들어있는 폴더 경로를 지정
    glob='*.txt',  # 폴더 내의 모든 파일이 아니라, 텍스트 파일(.txt)만 골라서 가져오도록 필터링합니다.
    # (중요) 각 파일을 실제로 읽을 때 사용할 '일꾼' 클래스를 지정합니다. 
    # 여기서는 일반 텍스트 읽기 도구를 선택했습니다.
    loader_cls=TextLoader, 
    # (핵심) 일꾼(TextLoader)에게 전달할 추가 옵션입니다. 
    # 한글이 깨지지 않도록 utf-8 방식으로 읽으라고 명확히 지시합니다.
    loader_kwargs={'encoding': 'utf-8'}
)

# 설정된 규칙에 따라 폴더 내의 모든 파일을 읽어서 각각 Document 객체로 
# 만든 뒤, data라는 리스트에 담습니다.
data = loader.load()

len(data) # 불러온 문서가 총 몇 개인지 확인

data[0] # 불러온 문서 리스트 중 첫 번째 문서 객체의 내용을 출력
