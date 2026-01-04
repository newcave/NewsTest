import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from google import genai
from duckduckgo_search import DDGS

# --- 페이지 설정 ---
st.set_page_config(page_title="KIHS 뉴스 분석기", page_icon="💧", layout="wide")

st.title("💧 KIHS 지능형 뉴스 분석기")
st.caption("DuckDuckGo (한국어 최신) + GDELT (글로벌 빅데이터) 하이브리드 검색")

# --- API 키 설정 ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    # 로컬 테스트용 인풋
    api_key = st.text_input("Google API Key", type="password")

if not api_key:
    st.warning("Google API Key를 입력해야 분석 기능을 사용할 수 있습니다.")
    st.stop()

client = genai.Client(api_key=api_key)

# ============================================================
# 1. 수집기 함수 정의 (DDG & GDELT)
# ============================================================

def fetch_duckduckgo_news(query, max_results=30):
    """DuckDuckGo를 통한 한국어 뉴스 검색"""
    results = []
    try:
        with DDGS() as ddgs:
            # region='kr-kr'로 한국 언론 우선 검색
            ddg_gen = ddgs.news(
                keywords=query,
                region="kr-kr",
                safesearch="off",
                timelimit="y", # m(한달), w(일주일), y(일년)
                max_results=max_results
            )
            for r in ddg_gen:
                results.append({
                    "source_type": "DuckDuckGo(한국)",
                    "title": r.get('title'),
                    "url": r.get('url'),
                    "published": r.get('date'),
                    "source_name": r.get('source'),
                    "snippet": r.get('body'),
                })
    except Exception as e:
        st.error(f"DuckDuckGo 검색 오류: {e}")
    return results

def fetch_gdelt_data(query, days=90, max_records=100):
    """GDELT API를 통한 글로벌 데이터 검색"""
    # 날짜 계산 (YYYYMMDDHHMMSS 포맷)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y%m%d000000")
    end_str = end_date.strftime("%Y%m%d235959")
    
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    # GDELT는 영어 쿼리가 아니면 결과가 거의 없음 -> 팁: 번역해서 던지거나 영어 약어 사용
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "maxrecords": max_records,
        "sort": "datedesc"
    }
    
    results = []
    try:
        r = requests.get(base_url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            articles = data.get("articles", [])
            for a in articles:
                results.append({
                    "source_type": "GDELT(글로벌)",
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "published": a.get("seendate"), # 포맷 변환 필요할 수 있음
                    "source_name": a.get("domain"),
                    "snippet": "(GDELT는 요약을 제공하지 않음)",
                })
    except Exception as e:
        st.error(f"GDELT 검색 오류: {e}")
    return results

# ============================================================
# 2. 사이드바 UI (검색 설정)
# ============================================================
with st.sidebar:
    st.header("🔍 검색 설정")
    
    # 검색어 입력
    default_q = '"KIHS" OR "한국수자원조사기술원" OR "수자원공사"'
    query = st.text_area("검색어 입력", value=default_q, height=100)
    
    st.markdown("---")
    st.subheader("데이터 소스 선택")
    
    # 소스 선택 체크박스
    use_ddg = st.checkbox("DuckDuckGo (한국어 뉴스 추천)", value=True, help="네이버/구글 뉴스와 유사하게 한국어 기사를 잘 찾아줍니다.")
    use_gdelt = st.checkbox("GDELT (해외/빅데이터)", value=False, help="전세계 데이터를 찾습니다. 한글 검색어는 잘 안 잡힐 수 있습니다.")
    
    st.markdown("---")
    max_items = st.slider("수집 개수 (소스당)", 10, 100, 30)
    
    btn_run = st.button("뉴스 수집 및 분석 시작", type="primary")

# ============================================================
# 3. 메인 로직
# ============================================================
if btn_run:
    if not query:
        st.warning("검색어를 입력해주세요.")
        st.stop()
        
    if not (use_ddg or use_gdelt):
        st.warning("최소한 하나의 데이터 소스를 선택해야 합니다.")
        st.stop()

    all_data = []

    # 1) DuckDuckGo 실행
    if use_ddg:
        with st.spinner("🦆 DuckDuckGo에서 한국어 뉴스를 수집 중..."):
            ddg_res = fetch_duckduckgo_news(query, max_results=max_items)
            all_data.extend(ddg_res)
            
    # 2) GDELT 실행
    if use_gdelt:
        with st.spinner("🌍 GDELT에서 글로벌 데이터를 탐색 중..."):
            gdelt_res = fetch_gdelt_data(query, max_records=max_items)
            all_data.extend(gdelt_res)
            
    # 결과 확인
    if not all_data:
        st.error("검색 결과가 없습니다. 검색어를 변경하거나 소스를 확인해주세요.")
    else:
        df = pd.DataFrame(all_data)
        
        # 탭 UI 구성
        tab1, tab2 = st.tabs(["📊 AI 분석 보고서", "📋 수집 데이터 원본"])
        
        with tab1:
            st.markdown(f"### 🤖 Gemini 분석 결과 (총 {len(df)}건 기반)")
            
            # 프롬프트 생성 (데이터가 너무 많으면 잘라냄)
            # 토큰 절약을 위해 제목, 출처, 앞부분 요약만 가져감
            context_list = []
            for idx, row in df.iterrows():
                context_list.append(f"- [{row['source_type']}] {row['title']} ({row['source_name']}): {row['snippet'][:100]}...")
            
            context_text = "\n".join(context_list[:50]) # 최대 50개까지만 분석에 사용
            
            prompt = f"""
            당신은 수자원 분야 정책 분석가입니다. 
            아래 수집된 뉴스 데이터({len(df)}건)를 바탕으로 'KIHS(한국수자원조사기술원) 및 물관리 동향 보고서'를 작성해주세요.

            [수집된 뉴스 데이터]
            {context_text}

            [작성 양식]
            1. **핵심 이슈 요약** (3줄 이내, 명확하게)
            2. **주요 키워드** (해시태그 형태, 5개)
            3. **기관별 동향** (언급된 기관 위주로 기술, 긍정/부정 이슈 구분)
            4. **시사점 및 제언** (KIHS 입장에서의 대응 방안)
            
            * 주의: 없는 내용은 지어내지 말고, 수집된 데이터에 근거해서만 작성할 것.
            """
            
            with st.spinner("Gemini가 보고서를 작성하고 있습니다..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=prompt
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI 분석 중 오류 발생: {e}")

        with tab2:
            st.markdown("### 📋 수집된 뉴스 목록")
            st.dataframe(
                df[['source_type', 'title', 'published', 'source_name', 'url']],
                use_container_width=True,
                hide_index=True
            )
            
            # CSV 다운로드
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "CSV로 다운로드",
                csv,
                "news_analysis_result.csv",
                "text/csv",
                key='download-csv'
            )

else:
    # 초기 화면 안내
    st.info("👈 왼쪽 사이드바에서 검색어를 입력하고 '분석 시작'을 눌러주세요.")
    st.markdown("""
    ### 💡 사용 팁
    * **한국어 자료가 필요할 때:** `DuckDuckGo` 체크 (필수)
    * **해외 사례/영문 자료가 필요할 때:** `GDELT` 체크 + **영문 검색어** 입력 (예: `Water Management Korea`)
    * **검색어 예시:** `한국수자원조사기술원 OR 수자원공사 OR 홍수 예보`
    """)
