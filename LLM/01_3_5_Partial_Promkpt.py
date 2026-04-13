"""
01_3_5_Partial_Prompt
LangChain의 부분 프롬프트 구성(Partial Prompt Templates) 기능을 다루고 있습니다. 
모든 변수를 한꺼번에 넣는 대신, 일부 변수를 미리 채워두거나(고정값), 함수를 통해 
자동으로 채워지게(동적값) 만드는 기법입니다.
"""

### 1. 정적(Static) 부분 변수 지정: .partial() 사용

from langchain_core.prompts import PromptTemplate

# 'layer'와 'element' 두 개의 변수를 가진 기본 템플릿을 만듭니다.
prompt = PromptTemplate.from_template("지구의 {layer}에서 가장 흔한 원소는 {element}입니다.")

# .partial() 메서드를 사용해 'layer' 변수에 "지각"이라는 값을 미리 채워둡니다.
# 이제 이 프롬프트는 'element' 값만 기다리는 상태가 됩니다.
partial_prompt = prompt.partial(layer="지각")

# 마지막으로 남은 'element'에 "산소"를 넣어 최종 문장을 출력합니다.
print(partial_prompt.format(element="산소"))

### 2. 프롬프트 초기화 시 부분 변수 지정
# 객체를 생성할 때 partial_variables를 통해 'layer'를 "맨틀"로 고정합니다.
prompt = PromptTemplate(
    template="지구의 {layer}에서 가장 흔한 원소는 {element}입니다.",
    input_variables=["element"],  # 사용자가 나중에 입력해야 할 변수 목록
    partial_variables={"layer": "맨틀"}  # 시스템이 이미 알고 있는 변수
)

# 남은 'element' 변수만 입력하여 문장 생성. 사용자는 이제 'element' 값만 전달하면 됩니다.
print(prompt.format(element="규소"))

### 3. 함수를 이용한 동적(Dynamic) 부분 변수 지정
# 값이 고정된 것이 아니라, **코드가 실행되는 시점의 상황(시간, 날짜 등)**에 따라 
# 자동으로 변하게 만드는 고급 기법입니다.

from datetime import datetime

# 현재 월(month)을 확인하여 "봄, 여름, 가을, 겨울" 중 하나를 반환하는 함수를 정의합니다.
def get_current_season():
    month = datetime.now().month
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 9:
        return "여름"
    elif 10 <= month <= 11:
        return "가을"
    else:
        return "겨울"

# 프롬프트를 만들 때 'season' 변수에 위에서 만든 '함수 이름'을 전달합니다.
# 주의: 함수를 실행(get_current_season())하는 게 아니라 함수 자체를 등록하는 것입니다.
prompt = PromptTemplate(
    template="{season}에 일어나는 대표적인 지구과학 현상은 {phenomenon}입니다.",
    input_variables=["phenomenon"],  # 사용자 입력이 필요한 변수
    partial_variables={"season": get_current_season} # 실행 시점에 함수가 호출되어 값이 채워짐
)

# 'phenomenon'만 입력하면, 'season'은 실행 시점의 실제 계절로 자동 계산됩니다.
print(prompt.format(phenomenon="개나리 개화")) 