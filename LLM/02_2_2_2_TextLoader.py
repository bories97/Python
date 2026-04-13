from langchain_community.document_loaders import TextLoader

# encoding='utf-8'을 추가하여 한글 파일을 정상적으로 읽게 합니다.윈도우용
loader = TextLoader("./data/history.txt", encoding='utf-8')

data = loader.load()

print(type(data))
print(len(data))

data

len(data[0].page_content)

data[0].metadata