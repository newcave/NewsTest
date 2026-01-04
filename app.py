import re
import json
import time
import hashlib
from datetime import datetime
from urllib.parse import quote

import pandas as pd
import requests
import feedparser
import streamlit as st
from dateutil.relativedelta import relativedelta
from google import genai
from duckduckgo_search import DDGS  # [NEW] 라이브러리 추가

# ============================================================
# KIHS (한국수자원조사기술원) 온라인 데이터 분석기 (Pro)
# - Sources: GDELT(대량) + Google RSS(속보) + DuckDuckGo(한글정확도)
# - Quarter UI (분기/브랜치)
# - Gemini 분석 보고서 Pool(누적 저장)
# ============================================================

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="KIHS 온라인 데이터 분석기 (Pro)", layout="wide")
st.title("KIHS (한국수자원조사기술원)")
st.caption("온라인 데이터 분석기 (Pro) — GDELT + DuckDuckGo + Google RSS + Gemini")

# -------------------------
# Secrets / Gemini
# -------------------------
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다. (Streamlit Cloud → Settings → Secrets)")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("❌ Gemini Client 생성 실패")
    st.exception(e)
    st.stop()

# -------------------------
# Session state (report pool)
# -------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "quarters" not in st.session_state:
    st.session_state["quarters"] = []
if "report_pool" not in st.session_state:
    st.session_state["report_pool"] = {}

# -------------------------
# Quarter utilities
# -------------------------
def parse_quarter(qstr: str):
    # "2024-Q1" -> (start, end_exclusive)
    y, q = qstr.split("-Q")
    y, q = int(y), int(q)
    start_month = (q - 1) * 3 + 1
    start = datetime(y, start_month, 1)
    end = start + relativedelta(months=3)
    return start, end

def quarter_label(dt: datetime) -> str:
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{q}"

def quarter_iter(start_q: str, end_q: str):
    # inclusive labels
    s, _ = parse_quarter(start_q)
    _, end_excl = parse_quarter(end_q)
    cur = s
    while cur < end_excl:
        qlab = quarter_label(cur)
        nxt = cur + relativedelta(months=3)
        yield qlab, cur, nxt
        cur = nxt

def dt_to_gdelt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")

def safe_iso_from_gdelt(seendate: str):
    # "YYYYMMDDHHMMSS" -> iso
    if isinstance(seendate, str) and re.fullmatch(r"\d{14}", seendate):
        try:
            return datetime.strptime(seendate, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            return None
    return None

# -------------------------
# Collectors (cached)
# -------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_gdelt_doc(query: str, start_dt: datetime, end_dt: datetime, max_records: int = 250):
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    out = []
    startrecord = 0  # GDELT API 조정
    pagesize = min(250, max_records)

    # 한글 검색 팁: GDELT는 한국어 쿼리가 약하므로 sourcecountry:KS 추가 고려 가능하나
    # 여기서는 원본 로직 유지하되 에러 처리 강화
    
    while len(out) < max_records:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "startdatetime": dt_to_gdelt(start_dt),
            "enddatetime": dt_to_gdelt(end_dt),
            "maxrecords": min(pagesize, max_records - len(out)),
            "startrecord": startrecord,
            "sort": "datedesc",
        }
        try:
            r = requests.get(base, params=params, timeout=10)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break

        articles = data.get("articles") or []
        if not articles:
            break

        for a in articles:
            out.append(
                {
                    "source_system": "GDELT",
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "published": safe_iso_from_gdelt(a.get("seendate")),
                    "snippet": a.get("snippet", ""),
                    "source": a.get("domain"),
                }
            )

        fetched = len(articles)
        startrecord += fetched
        if fetched < params.get("maxrecords", 250):
            break

        time.sleep(0.15) 

    return out

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_google_news_rss(query: str, hl="ko", gl="KR", ceid="KR:ko", limit=80):
    q = quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    d = feedparser.parse(url)

    out = []
    for e in (d.entries or [])[:limit]:
        published_iso = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            try:
                published_iso = datetime(*e.published_parsed[:6]).isoformat()
            except Exception:
                published_iso = None

        src = None
        if hasattr(e, "source"):
            try:
                src = e.source.get("title")
            except Exception:
                src = None

        out.append(
            {
                "source_system": "GoogleRSS",
                "title": getattr(e, "title", None),
                "url": getattr(e, "link", None),
                "domain": "news.google.com",
                "language": "ko",
                "published": published_iso,
                "snippet": getattr(e, "summary", "") or "",
                "source": src,
            }
        )
    return out

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_duckduckgo_news(query: str, max_results: int = 50):
    """
    [NEW] DuckDuckGo 수집기 추가
    한국어 키워드에 매우 강력하며, 본문 요약(snippet) 품질이 좋습니다.
    """
    out = []
    try:
        with DDGS() as ddgs:
            # region='kr-kr'로 한국 언론사 우선 검색
            ddg_gen = ddgs.news(
                keywords=query,
                region="kr-kr",
                safesearch="off",
                timelimit="y",  # 최근 1년치 검색 (이후 로직에서 분기별로 필터링됨)
                max_results=max_results
            )
            
            for r in ddg_gen:
                # 날짜 파싱 시도 (DDG는 ISO 비슷한 포맷으로 줌)
                pub_iso = None
                raw_date = r.get('date')
                if raw_date:
                    try:
                        # 2024-05-20T14:00:00+00:00 형식이 일반적
                        dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                        pub_iso = dt.isoformat()
                    except:
                        pub_iso = None # 날짜 파싱 실패시 None 처리 (이후 로직에서 걸러짐)

                out.append({
                    "source_system": "DuckDuckGo",
                    "title": r.get('title'),
                    "url": r.get('url'),
                    "domain": r.get('source'), # DDG는 source 필드에 언론사명 제공
                    "language": "ko",
                    "published": pub_iso,
                    "snippet": r.get('body', ''),
                    "source": r.get('source'),
                })
    except Exception as e:
        # st.error(f"DDG Error: {e}") # 사용자에게 에러 노출 최소화
        pass
        
    return out

# -------------------------
# Normalization / dedup / sentiment
# -------------------------
POS_WORDS = ["확대", "성장", "도입", "개선", "성과", "혁신", "지원", "투자", "상용화", "성공", "협력", "발전", "체결", "달성"]
NEG_WORDS = ["우려", "논란", "실패", "중단", "규제", "사고", "부족", "지연", "위험", "갈등", "반대", "피해", "적발", "오염"]

def rule_sentiment(text: str) -> str:
    t = (text or "").lower()
    p = sum(w in t for w in POS_WORDS)
    n = sum(w in t for w in NEG_WORDS)
    if p > n:
        return "긍정"
    if n > p:
        return "부정"
    return "중립"

def make_key(url: str, title: str) -> str:
    base = (url or "").strip() or (title or "").strip()
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()

def query_fingerprint(query: str, start_q: str, end_q: str, use_gdelt: bool, use_rss: bool, use_ddg: bool, gdelt_max: int, rss_max: int, ddg_max: int):
    s = json.dumps(
        {
            "query": query,
            "start_q": start_q,
            "end_q": end_q,
            "use_gdelt": use_gdelt,
            "use_rss": use_rss,
            "use_ddg": use_ddg,
            "gdelt_max": gdelt_max,
            "rss_max": rss_max,
            "ddg_max": ddg_max
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# -------------------------
# Gemini reporting
# -------------------------
def build_quarter_bullets(dfq: pd.DataFrame, cap: int = 150) -> str:
    d = dfq.copy()
    d["published_dt"] = pd.to_datetime(d["published"], errors="coerce")
    d = d.sort_values("published_dt", ascending=False).drop(columns=["published_dt"])

    lines = []
    for _, r in d.head(cap).iterrows():
        # Source System을 앞에 표기하여 출처 구분
        src = r.get("domain") or r.get("source") or "Unknown"
        lines.append(f"- [{r['source_system']}/{r['sentiment']}] {r['title']} ({src})")
    return "\n".join(lines)

def gemini_report(quarter: str, bullets: str, model_name: str):
    prompt = f"""
당신은 'KIHS (한국수자원조사기술원) 지능형 데이터 분석기'의 수석 분석가입니다.

대상 분기: {quarter}

아래는 GDELT(글로벌), DuckDuckGo(한국어), Google RSS에서 수집된 뉴스 헤드라인입니다.
이 데이터를 종합하여 과장 없이 전문가 수준의 동향 보고서를 작성하세요.

[수집된 뉴스 데이터]
{bullets}

[보고서 작성 양식]
1. **분기 총평 및 핵심 이슈** (3~5문장 요약)
2. **주요 기관/기업 동향** - KIHS, K-water, 환경부 등 주요 주체별 활동 정리
3. **분야별 이슈 분석** (긍정/부정)
   - [긍정/성과] 기술 도입, 협약 체결 등
   - [부정/리스크] 가뭄/홍수 피해, 갈등, 사고 등
4. **KIHS를 위한 시사점** (실행 가능한 제언 3가지)
5. **차기 모니터링 키워드** (해시태그 5개)

제약사항:
- 데이터에 없는 내용은 절대 지어내지 말 것.
- 'DuckDuckGo', 'GDELT' 등의 시스템 용어는 보고서 본문에 쓰지 말 것.
- 문체는 "함.", "됨." 등의 개조식이 아닌, 정중한 보고서체("하였습니다.")를 사용할 것.
"""
    res = client.models.generate_content(model=model_name, contents=prompt)
    return res.text or ""

# -------------------------
# Sidebar UI
# -------------------------
with st.sidebar:
    st.header("설정")

    st.subheader("기간(분기) 선택")
    c1, c2 = st.columns(2)
    with c1:
        start_q = st.text_input("시작 분기", value="2024-Q1")
    with c2:
        end_q = st.text_input("종료 분기", value="2025-Q1")

    st.subheader("검색어(Query)")
    # 한국어 검색을 위해 키워드 보강
    default_query = (
        '"KIHS" OR "한국수자원조사기술원" OR "수자원공사" OR "환경부 물관리" OR '
        'flood OR drought OR "smart water" OR "digital twin"'
    )
    query = st.text_area("검색어", value=default_query, height=120)

    st.subheader("데이터 소스 설정")
    
    # 1. DuckDuckGo (New, Korean Strong)
    use_ddg = st.checkbox("DuckDuckGo (한국어 추천)", value=True, help="한국어 뉴스 검색 정확도가 높습니다.")
    ddg_max = st.slider("DDG 수집량 (소스당)", 30, 200, 100, 10)
    
    # 2. RSS (Supplement)
    use_rss = st.checkbox("Google News RSS (보강용)", value=True)
    rss_max = st.slider("RSS 수집량 (전체)", 20, 200, 80, 10)

    # 3. GDELT (Global, BigData)
    use_gdelt = st.checkbox("GDELT (해외/빅데이터)", value=True, help="한글 검색은 약하지만, 영문/글로벌 추세 파악에 필수적입니다.")
    gdelt_max = st.slider("GDELT 수집량 (분기당)", 50, 1000, 250, 50)
    
    st.markdown("---")
    st.subheader("LLM 설정")
    model_name = st.selectbox("모델", ["gemini-2.5-flash", "gemini-1.5-pro"], index=0)

    st.markdown("---")
    btn_collect = st.button("① 데이터 수집 및 전처리", type="primary")
    btn_analyze = st.button("② AI 보고서 생성", type="secondary")
    btn_clear_pool = st.button("초기화 (Reset)", type="tertiary")

# -------------------------
# Actions
# -------------------------
if btn_clear_pool:
    st.session_state["report_pool"] = {}
    st.session_state["df"] = None
    st.session_state["summary"] = None
    st.success("모든 데이터를 초기화했습니다.")

def run_collection():
    # validate quarters
    try:
        quarters = list(quarter_iter(start_q, end_q))
        if not quarters:
            raise ValueError("empty")
    except Exception:
        st.error("분기 형식이 올바르지 않습니다. 예: 2024-Q1")
        return

    all_rows = []
    
    # 1. Collect GDELT (분기별 루프)
    if use_gdelt:
        with st.spinner("🌍 GDELT(글로벌) 데이터 수집 중..."):
            for qlab, qs, qe in quarters:
                recs = fetch_gdelt_doc(query, qs, qe, max_records=gdelt_max)
                for r in recs:
                    r["quarter"] = qlab
                    all_rows.append(r)

    # 2. Collect RSS (전체 기간 -> 날짜기반 할당)
    if use_rss:
        with st.spinner("📰 Google News RSS 데이터 수집 중..."):
            rss_recs = fetch_google_news_rss(query, limit=rss_max)
            for r in rss_recs:
                pub = r.get("published")
                if not pub: continue
                try:
                    dt = datetime.fromisoformat(pub)
                    r["quarter"] = quarter_label(dt)
                    all_rows.append(r)
                except:
                    continue

    # 3. Collect DuckDuckGo (전체 기간 -> 날짜기반 할당) [NEW]
    if use_ddg:
        with st.spinner("🦆 DuckDuckGo(한국어) 데이터 수집 중..."):
            ddg_recs = fetch_duckduckgo_news(query, max_results=ddg_max)
            for r in ddg_recs:
                pub = r.get("published")
                if not pub: continue
                try:
                    dt = datetime.fromisoformat(pub)
                    r["quarter"] = quarter_label(dt)
                    all_rows.append(r)
                except:
                    # 날짜 형식이 안 맞으면 현재 분기 혹은 제외 처리
                    continue

    if not all_rows:
        st.warning("수집 결과가 없습니다. 검색어/기간/소스를 조정해 주세요.")
        return

    df = pd.DataFrame(all_rows)

    # normalize cols
    needed = ["title", "url", "published", "source_system", "quarter", "domain", "language", "snippet", "source"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    # dedup
    df["key"] = [make_key(u, t) for u, t in zip(df["url"].astype(str), df["title"].astype(str))]
    df = df.drop_duplicates(subset=["key"]).copy()

    # strictly filter quarters within selected range
    # (RSS와 DDG는 최근 데이터를 가져오므로, 사용자가 선택한 분기 범위를 벗어날 수 있음 -> 필터링)
    wanted_quarters = [q for q, _, _ in quarters]
    df = df[df["quarter"].isin(wanted_quarters)].copy()

    # sentiment
    df["sentiment"] = df["title"].apply(rule_sentiment)

    # sort
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.sort_values(["quarter", "published_dt"], ascending=[True, False]).drop(columns=["published_dt"])

    # summary table
    summary = (
        df.groupby(["quarter", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="quarter", columns="sentiment", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    st.session_state["df"] = df
    st.session_state["summary"] = summary
    st.session_state["quarters"] = wanted_quarters

    st.success(f"✅ 수집 및 통합 완료: 총 {len(df):,}건 (GDELT+RSS+DDG)")

if btn_collect:
    run_collection()

# -------------------------
# Main view
# -------------------------
df = st.session_state.get("df")
summary = st.session_state.get("summary")
quarters = st.session_state.get("quarters", [])
report_pool = st.session_state.get("report_pool", {})

if df is None or summary is None:
    st.info("좌측 사이드바에서 **① 데이터 수집 및 전처리** 버튼을 눌러주세요.")
    st.stop()

# Layout
left, right = st.columns([1.3, 1.0], gap="large")

with left:
    st.subheader("🗂️ 수집 데이터 필터링")
    st.write(f"통합 데이터베이스: **{len(df):,}건**")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        quarter_sel = st.selectbox("분기 선택", sorted(df["quarter"].unique()))
    with f2:
        source_sel = st.selectbox("출처", ["전체"] + sorted(list(df["source_system"].dropna().unique())))
    with f3:
        sentiment_sel = st.selectbox("감성", ["전체", "긍정", "중립", "부정"], index=0)
    with f4:
        kw = st.text_input("내용 검색", value="", placeholder="키워드 입력")

    # Filter Logic
    dff = df[df["quarter"] == quarter_sel].copy()
    if source_sel != "전체":
        dff = dff[dff["source_system"] == source_sel]
    if sentiment_sel != "전체":
        dff = dff[dff["sentiment"] == sentiment_sel]
    if kw.strip():
        mask = dff["title"].str.contains(kw, case=False, na=False) | dff["snippet"].str.contains(kw, case=False, na=False)
        dff = dff[mask]

    show_cols = ["published", "source_system", "sentiment", "title", "domain", "url"]
    st.dataframe(dff[show_cols], use_container_width=True, height=450, hide_index=True)

    csv_bytes = dff[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 현재 필터 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"KIHS_{quarter_sel}_filtered.csv",
        mime="text/csv",
    )

with right:
    st.subheader("📊 분기별 데이터 요약")
    st.dataframe(summary, use_container_width=True, height=150, hide_index=True)

    chart_df = summary.set_index("quarter")
    for col in ["긍정", "중립", "부정"]:
        if col not in chart_df.columns:
            chart_df[col] = 0
    st.bar_chart(chart_df[["긍정", "중립", "부정"]], height=200, stack=True)

    st.markdown("---")
    st.subheader("📝 AI 분석 리포트 (Report Pool)")
    
    # pool overview
    if report_pool:
        pool_list = []
        for q, meta in report_pool.items():
            pool_list.append({
                "분기": q,
                "작성시각": meta.get("created_at", "").split("T")[1][:5], # 시간만 표시
                "기사수": meta.get("n_items", 0),
            })
        st.dataframe(pd.DataFrame(pool_list).sort_values("분기"), use_container_width=True, height=120, hide_index=True)
    else:
        st.info("생성된 보고서가 없습니다. 하단에서 생성하세요.")

    st.markdown("#### 보고서 생성 및 열람")
    gen_targets = st.multiselect("분석 대상 분기 선택", quarters_list, default=[quarter_sel])

    # Analyze action
    if btn_analyze:
        # Use simple hash for demo
        fp = query_fingerprint(query, start_q, end_q, use_gdelt, use_rss, use_ddg, gdelt_max, rss_max, ddg_max)
        
        with st.spinner("Gemini가 데이터를 읽고 보고서를 작성 중입니다..."):
            new_pool = dict(report_pool)
            for q in gen_targets:
                dfq = df[df["quarter"] == q]
                if dfq.empty:
                    continue
                    
                # 프롬프트에 들어갈 뉴스 리스트 생성
                bullets = build_quarter_bullets(dfq, cap=100) # 토큰 제한 고려
                
                try:
                    text = gemini_report(q, bullets, model_name=model_name)
                    new_pool[q] = {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "model": model_name,
                        "n_items": int(len(dfq)),
                        "query_hash": fp,
                        "text": text,
                    }
                except Exception as e:
                    st.error(f"{q} 분석 중 오류: {e}")
            
            st.session_state["report_pool"] = new_pool
            report_pool = new_pool
        st.success("보고서 작성이 완료되었습니다!")

    # View report
    if report_pool:
        view_q = st.selectbox("열람할 보고서 분기", options=sorted(report_pool.keys()), key="view_q")
        with st.expander(f"{view_q} 보고서 보기", expanded=True):
            st.markdown(report_pool[view_q].get("text", ""))

        st.download_button(
            label="📄 전체 리포트 풀(JSON) 저장",
            data=json.dumps(report_pool, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="KIHS_report_pool.json",
            mime="application/json",
        )
