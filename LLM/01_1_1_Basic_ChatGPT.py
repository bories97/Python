from openai import OpenAI
import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")
# 클라이언트를 초기화합니다. 
# api_key를 직접 넣거나, 환경 변수 OPENAI_API_KEY에 키를 저장해두면 자동으로 불러옵니다.
client = OpenAI(
    api_key="OPENAI_API_KEY"
)

# Chat Completions API를 호출합니다.
response = client.chat.completions.create(
    model="gpt-4o",  # 또는 "gpt-4o-mini" (저렴하고 빠름)
    messages=[
        {"role": "user", "content": "지구의 자전 주기는?"}
    ]
)

# 결과 출력
print(response.choices[0].message.content)