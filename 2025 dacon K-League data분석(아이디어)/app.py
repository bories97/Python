# app.py
import time
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# =========================
# 0. 페이지 설정
# =========================
st.set_page_config(
    page_title="Live Match Replay",
    layout="centered"
)

st.title("⚽ 실시간 경기 리플레이 (모델 결과 기반)")

# =========================
# 1. session_state 초기화
# =========================
if "playing" not in st.session_state:
    st.session_state.playing = False

if "time_idx" not in st.session_state:
    st.session_state.time_idx = 0

# =========================
# 2. 데이터 로드
# =========================
@st.cache_data
def load_data():
    url_match = "https://raw.githubusercontent.com/bories97/Python/refs/heads/main/%EA%B3%B5%EB%AA%A8%EC%A0%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0/match_info.csv"
    url_state = "https://raw.githubusercontent.com/bories97/Python/refs/heads/main/%EA%B3%B5%EB%AA%A8%EC%A0%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0/state_df.csv"
    url_demo  = "https://raw.githubusercontent.com/bories97/Python/refs/heads/main/%EA%B3%B5%EB%AA%A8%EC%A0%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0/demo_df.csv"

    match = pd.read_csv(url_match)
    states = pd.read_csv(url_state)
    demo = pd.read_csv(url_demo)

    return match, states, demo

match, states, demo = load_data()

# =========================
# 3. 경기 선택
# =========================
available_games = (
    match[['game_id', 'home_team_name_ko', 'away_team_name_ko']]
    .drop_duplicates()
    .sort_values('game_id')
)

labels = (
    available_games['game_id'].astype(str)
    + " | "
    + available_games['home_team_name_ko']
    + " vs "
    + available_games['away_team_name_ko']
)

selected_label = st.selectbox("경기 선택", labels)
SELECTED_GAME_ID = int(selected_label.split(" | ")[0])

home_name = available_games.loc[
    available_games['game_id'] == SELECTED_GAME_ID,
    'home_team_name_ko'
].values[0]

away_name = available_games.loc[
    available_games['game_id'] == SELECTED_GAME_ID,
    'away_team_name_ko'
].values[0]

# =========================
# 4. 시각화용 데이터
# =========================
viz_df = (
    states[states['game_id'] == SELECTED_GAME_ID]
    .sort_values('time_bin')
    .reset_index(drop=True)
)

# =========================
# 5. 컨트롤 버튼
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Play"):
        st.session_state.playing = True

with col2:
    if st.button("⏸ Pause"):
        st.session_state.playing = False

# =========================
# 6. 시간 슬라이더 (충돌 방지)
# =========================
if not st.session_state.playing:
    st.session_state.time_idx = st.slider(
        "⏱ 경기 시간",
        min_value=0,
        max_value=len(viz_df) - 1,
        value=st.session_state.time_idx
    )
else:
    st.slider(
        "⏱ 경기 시간",
        min_value=0,
        max_value=len(viz_df) - 1,
        value=st.session_state.time_idx,
        disabled=True
    )

row = viz_df.iloc[st.session_state.time_idx]
current_time = int(row["time_bin"])

# =========================
# 7. 실제 스코어
# =========================
score_row = (
    demo[(demo["game_id"] == SELECTED_GAME_ID) &
         (demo["time_bin"] <= current_time)]
    .sort_values("time_bin")
)

if len(score_row) > 0:
    last_score = score_row.iloc[-1]
    h_score = int(last_score["home_score"])
    a_score = int(last_score["away_score"])
else:
    h_score, a_score = 0, 0

# =========================
# 8. 확률 & 해설
# =========================
p_h = row["p_home"] * 100
p_d = row["p_draw"] * 100
p_a = row["p_away"] * 100

momentum = (
    0.15 * row["home_xgdiff"]
    + 0.10 * row["home_xadiff"]
    + 0.05 * row["home_xtdiff"]
)

if abs(momentum) > 0.02:
    team = home_name if momentum > 0 else away_name
    color = "#e03e2d" if momentum > 0 else "#2d6ae0"
    commentary = f"<b style='color:{color};'>{team} 공격 흐름 우세</b>"
elif row["remain_time"] < 600 and row["score_diff"] != 0:
    commentary = "<b>경기 막판 굳히기 국면</b>"
else:
    commentary = "<span style='color:#aaa;'>탐색전 진행 중</span>"

# =========================
# 9. HTML 카드 출력
# =========================
html_code = f"""
<div style="
    width:600px; margin:20px auto; font-family:Arial;
    border:1px solid #e1e4e8; border-radius:12px;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
">
    <div style="background:#f6f8fa; padding:10px; text-align:center;">
        ⏱ Match Time: {current_time} min
    </div>

    <div style="padding:25px 0; display:flex; justify-content:space-around;">
        <div style="width:35%; text-align:center; font-weight:bold;">{home_name}</div>
        <div style="width:30%; text-align:center; font-size:40px;">
            <span style="color:#e03e2d;">{h_score}</span> :
            <span style="color:#2d6ae0;">{a_score}</span>
        </div>
        <div style="width:35%; text-align:center; font-weight:bold;">{away_name}</div>
    </div>

    <div style="padding:0 30px 25px;">
        <div style="display:flex; justify-content:space-between; font-size:12px;">
            <span>Home {p_h:.1f}%</span>
            <span>Draw {p_d:.1f}%</span>
            <span>Away {p_a:.1f}%</span>
        </div>
        <div style="display:flex; height:10px; background:#eee;">
            <div style="width:{p_h}%; background:#e03e2d;"></div>
            <div style="width:{p_d}%; background:#999;"></div>
            <div style="width:{p_a}%; background:#2d6ae0;"></div>
        </div>
    </div>

    <div style="background:#f6f8fa; padding:12px; text-align:center;">
        {commentary}
    </div>
</div>
"""

card_container = st.container(
    key=f"card_{st.session_state.time_idx}"
)

with card_container:
    components.html(html_code, height=420)

# =========================
# 10. 자동 재생 트리거 (⚠️ 반드시 맨 마지막)
# =========================
if st.session_state.playing:
    time.sleep(0.3)
    st.session_state.time_idx += 1

    if st.session_state.time_idx >= len(viz_df):
        st.session_state.time_idx = len(viz_df) - 1
        st.session_state.playing = False
    else:
        st.rerun()