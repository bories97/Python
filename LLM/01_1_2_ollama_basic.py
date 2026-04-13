# pip install -U langchain-community
# pip install -U langchain-ollama
# ollama pull gemma3:4b   # 가상환경에 설치해서 사용

# Ollama 라이브러리에서 Ollama 클래스를 불러옵니다.
# from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama


# 'gemma3:4b' 모델을 사용하여 llm 객체를 생성합니다.
llm = ChatOllama(model="gemma3:4b")

# 단순 질문을 던져서 모델의 응답을 받습니다. (invoke 메서드 사용)
result = llm.invoke("지구의 자전 주기는?")

result

# ============================================================
# 실습: Ollama 기초 (ollama_basic.py)
# ============================================================
# [필수 설치]
#   pip install openai python-dotenv
#   (Ollama 앱 별도 설치: https://ollama.ai)
#
# [사전 준비]
#   - 별도 CMD 창에서 'ollama serve' 실행 중. 기존 올라마 종료후 적용
#   - 모델 다운로드: 'ollama pull mistral:7b'
#
# [실행 방법]
#   python ollama_basic.py
#
# [예상 출력]
#   Ollama 모델: mistral:7b
#   질문: "Python이란?"
#   응답: "Python은 프로그래밍 언어입니다..." (2~10초 소요)
#
# [주의사항]
#   - Ollama 서버가 실행 중이어야 함 (11434 포트)
#   - 첫 실행은 모델 로드로 느림 (10~30초)
#   - GPU 있으면 5배 빠름 (CPU: 2~10초, GPU: 0.5~2초)
# ============================================================
from openai import OpenAI
from dotenv import load_dotenv
import time
import os

load_dotenv()

# (1) Ollama 클라이언트를 생성합니다 (OpenAI SDK 사용)
#     주의: api_key는 "ollama" 더미값 사용 (실제 필요 없음)
#     base_url: Ollama 서버 주소 (기본 localhost:11434)
client = OpenAI(
    api_key="ollama",                    # 더미 값 (Ollama는 API 키 불필요)
    base_url="http://localhost:11434/v1" # Ollama의 OpenAI 호환 엔드포인트
)

# (2) 사용할 모델 선택
#     mistral:7b: 가장 인기 있는 소형 모델, 빠르고 정확
#     다른 옵션: llama2:7b, neural-chat:7b, orca-mini:3b, gemma3:12b
model_name = "gemma3:4b"

print("=" * 60)
print("Ollama 기초 - Local LLM 사용하기")
print("=" * 60)
print(f"모델: {model_name}")
print(f"주소: http://localhost:11434")
print("=" * 60 + "\n")

# (3) 테스트 질문들을 정의합니다
test_questions = [
    "Python이란 무엇인가요?",
    "AI와 Machine Learning의 차이는?",
    "좋은 코드의 특징은?"
]

# (4) 각 질문에 대해 Ollama로 응답을 받습니다
for i, question in enumerate(test_questions, 1):
    print(f"[질문 {i}] {question}")
    
    # (5) Ollama API 호출 (OpenAI SDK와 동일한 방식)
    #     temperature: 창의성 (낮을수록 정확, 높을수록 창의적)
    #     max_tokens: 최대 응답 길이
    start_time = time.time()  # 응답 시간 측정용
    
    try:
        response = client.chat.completions.create(
            model=model_name,                    # Ollama 모델명
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7,                     # 중간 정도의 창의성
            max_tokens=300,                      # 300 토큰 이하로 제한
            # stream=False가 기본값이므로 전체 응답을 한 번에 받음
        )
        
        elapsed_time = time.time() - start_time  # 응답 시간 계산
        
        # (6) 응답을 추출하고 출력합니다
        answer = response.choices[0].message.content
        print(f"[응답 ({elapsed_time:.2f}초)]")
        print(f"{answer}\n")
        
    except Exception as e:
        # (7) 오류 처리
        print(f"[오류] {e}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        print("CMD에서 'ollama serve' 입력\n")

# (8) 스트리밍 예제 (선택사항)
print("=" * 60)
print("스트리밍 응답 예제")
print("=" * 60)

streaming_question = "Python 리스트와 딕셔너리의 차이를 설명해주세요."
print(f"질문: {streaming_question}\n")

try:
    # (9) stream=True로 설정하면 청크 단위로 응답 받음
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": streaming_question}],
        temperature=0.7,
        max_tokens=300,
        stream=True  # ★ 스트리밍 활성화
    )
    
    # (10) 청크를 하나씩 처리합니다 (실시간 출력)
    print("응답: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            # delta.content는 이 청크의 텍스트 조각
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
except Exception as e:
    print(f"[오류] {e}\n")

print("=" * 60)
print("Ollama 기초 테스트 완료")
print("=" * 60)