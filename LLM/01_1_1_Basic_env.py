import os
# pip install python-dotenv
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("안녕! 너에대해 짧게 소개해줘.")

print(result.content)