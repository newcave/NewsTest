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


# ============================================================
# KIHS (한국수자원조사기술원) 온라인 데이터 분석기 (데모)
# Korea Institute of Hydraulic Survey
# Sources: GDELT + Google News RSS
# LLM: Gemini
# ============================================================

st.set_page_config(
    page_title="KIHS 온라인 데이터 분석기 (데모)",
    layout="wide",
)

st.title("KIHS (한국수자원조사기술원)")
st.caption("온라인 데이터 분석기 (데모) — GDELT + Google News RSS + Gemini")


# =========================
# Secrets / API
# =========================
API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. Streamlit Cloud → Settings → Secrets에 GOOGLE_API_KEY를 추가하세요.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# =========================
# Quarter Utilities
# =========================
def parse_quarter(qstr: str):
    """
    qstr: "2024-Q1"
    returns: (start_datetime, end_datetime_exclusive)
    """
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
    """
    Iterate quarters from start_q (inclusive) to end_q (inclusive label).
    Internally, end boundary = end of end_q (exclusive).
    """
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
    # GDELT seendate is often "YYYYMMDDHHMMSS"
    if isinstance(seendate, str) and re.fullmatch(r"\d{14}", seendate):
        try:
            return datetime.strptime(seendate, "%Y%m%d%H%M%S").isoformat()
        except Exception:
            return None
    return None


# =========================
# Collectors
# =========================
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_gdelt_doc(query: str, start_dt: datetime, end_dt: datetime, max_records: int = 250):
    """
    GDELT 2.1 DOC API (artlist mode)
    """
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    out = []
    startrecord = 1
    pagesize = min(250, max_records)

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
            r = requests.get(base, params=params, timeout=30)
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
                    "snippet": a.get("snippet"),
                }
            )

        fetched = len(articles)
        startrecord += fetched
        if fetched < params["maxrecords"]:
            break

        time.sleep(0.2)  # 과도한 호출 방지

    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_google_news_rss(query: str, hl: str = "ko", gl: str = "KR", ceid: str = "KR:ko", limit: int = 100):
    """
    Google News RSS (보강용)
    - 피드 포맷/정책/제한이 변경될 수 있음
    - 정확한 커버리지를 보장하지 않음
    """
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
                "source_system": "GoogleNewsRSS",
                "title": getattr(e, "title", None),
                "url": getattr(e, "link", None),
                "domain": None,
                "language": None,
                "published": published_iso,
                "snippet": getattr(e, "summary", "") or "",
                "source": src,
            }
        )

    return out


# =========================
# Normalization / Dedup / Sentiment (Demo Rule)
# =========================
POS_WORDS = ["확대", "성장", "도입", "개선", "성과", "혁신", "지원", "투자", "상용화", "성공", "협력", "발전"]
NEG_WORDS = ["우려", "논란", "실패", "중단", "규제", "사고", "부족", "지연", "위험", "갈등", "반대", "피해"]


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


# =========================
# Gemini: Implications
# =========================
def gemini_implications(quarter: str, bullets: str, model_name: str):
    prompt = f"""
당신은 다음 시스템의 분석가입니다.

시스템명:
"KIHS (한국수자원조사기술원) 온라인 데이터 분석기 (데모)"
영문:
"Korea Institute of Hydraulic Survey Online Data Analyzer (Demo)"

대상 분기: {quarter}

아래 뉴스/기사 제목 목록을 바탕으로, 한국어로 '간결하고 단정한' 보고서를 작성하세요.
(가능하면 과장 없이, 실행 가능한 시사점 위주)

[입력 목록]
{bullets}

[출력 형식]
1) 분기 핵심 요약 (6줄 이내)
2) 긍정 요인 (bullet)
3) 부정 요인 (bullet)
4) 향후 정책 시사점 (3~6개 bullet, 실행 가능하게)
5) 향후 기술 시사점 (3~6개 bullet, 실행 가능하게)
6) 다음 분기 모니터링 키워드 (10개)
7) 전제/리스크 (bullet)
"""
    res = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return res.text


# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("설정")

    st.subheader("기간(분기) 선택")
    colA, colB = st.columns(2)
    with colA:
        start_q = st.text_input("시작 분기", value="2024-Q1")
    with colB:
        end_q = st.text_input("종료 분기", value="2025-Q1")

    st.subheader("검색어(Query)")
    default_query = (
        '"KIHS" OR "Korea Institute of Hydraulic Survey" OR "한국수자원조사기술원" OR '
        'hydrology OR flood OR drought OR dam OR reservoir OR "water treatment plant" OR wastewater OR leakage OR '
        '"digital twin" OR AI OR "numerical modeling" OR simulation'
    )
    query = st.text_area("GDELT/RSS 공통 검색어", value=default_query, height=140)

    st.subheader("데이터 소스")
    use_gdelt = st.checkbox("GDELT 사용(대량)", value=True)
    gdelt_max = st.slider("GDELT 분기당 최대 수집", 50, 1000, 250, 50)

    use_rss = st.checkbox("Google News RSS 사용(보강)", value=True)
    rss_max = st.slider("RSS 최대 수집(전체)", 20, 200, 80, 10)

    # RSS 위험/제약 경고
    st.markdown("### ⚠️ RSS 사용 시 주의")
    st.warning(
        "Google News RSS는 **보강용**입니다.\n"
        "- 피드 포맷/정책이 변경될 수 있음\n"
        "- 일부 기사만 노출(커버리지 제한)\n"
        "- 게시 시각/출처 메타데이터가 불완전할 수 있음\n"
        "- 대량 수집·상업적 활용은 약관/정책 이슈가 발생할 수 있음\n"
        "따라서 **기간 필터 정확도/대량 커버리지는 GDELT가 더 안정적**입니다."
    )

    st.subheader("LLM(Gemini)")
    model_name = st.selectbox("모델 선택", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)

    st.markdown("---")
    run_btn = st.button("① 수집/전처리 실행", type="primary")
    analyze_btn = st.button("② 분석(Analyze) 실행", type="secondary")


# =========================
# Pipeline
# =========================
def build_quarter_bullets(dfq: pd.DataFrame, cap: int = 120) -> str:
    d = dfq.copy()
    d["published_dt"] = pd.to_datetime(d["published"], errors="coerce")
    d = d.sort_values("published_dt", ascending=False).drop(columns=["published_dt"])

    lines = []
    for _, r in d.head(cap).iterrows():
        src = r.get("domain") or r.get("source") or r.get("source_system")
        lines.append(f"- [{r['sentiment']}] {r['title']} ({src})")
    return "\n".join(lines)


def run_collection(start_q: str, end_q: str, query: str, use_gdelt: bool, use_rss: bool, gdelt_max: int, rss_max: int):
    # validate quarters
    try:
        quarters = list(quarter_iter(start_q, end_q))
        if not quarters:
            raise ValueError("empty")
    except Exception:
        st.error("분기 형식이 올바르지 않습니다. 예: 2024-Q1")
        return None, None, None

    all_rows = []

    # GDELT
    if use_gdelt:
        with st.spinner("GDELT에서 분기별 수집 중..."):
            for qlabel, qs, qe in quarters:
                recs = fetch_gdelt_doc(query, qs, qe, max_records=gdelt_max)
                for r in recs:
                    r["quarter"] = qlabel
                    r["source"] = r.get("domain")
                    all_rows.append(r)

    # RSS (보강)
    if use_rss:
        with st.spinner("Google News RSS에서 보강 수집 중..."):
            rss_recs = fetch_google_news_rss(query, limit=rss_max)
            for r in rss_recs:
                pub = r.get("published")
                if not pub:
                    continue
                try:
                    dt = datetime.fromisoformat(pub)
                except Exception:
                    continue
                qlab = quarter_label(dt)
                r["quarter"] = qlab
                all_rows.append(r)

    if not all_rows:
        st.warning("수집 결과가 없습니다. 검색어 또는 기간/소스를 조정해 주세요.")
        return None, None, None

    df = pd.DataFrame(all_rows)

    # normalize cols
    for c in ["title", "url", "published", "source_system", "quarter", "domain", "language", "snippet", "source"]:
        if c not in df.columns:
            df[c] = None

    df["key"] = [make_key(u, t) for u, t in zip(df["url"].astype(str), df["title"].astype(str))]
    df = df.drop_duplicates(subset=["key"]).copy()

    wanted_quarters = [q for q, _, _ in quarters]
    df = df[df["quarter"].isin(wanted_quarters)].copy()

    df["sentiment"] = df["title"].apply(rule_sentiment)

    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.sort_values(["quarter", "published_dt"], ascending=[True, False]).drop(columns=["published_dt"])

    summary = (
        df.groupby(["quarter", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="quarter", columns="sentiment", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    return df, summary, wanted_quarters


# =========================
# Run
# =========================
if run_btn:
    df, summary, wanted_quarters = run_collection(start_q, end_q, query, use_gdelt, use_rss, gdelt_max, rss_max)
    st.session_state["df"] = df
    st.session_state["summary"] = summary
    st.session_state["quarters"] = wanted_quarters
    st.session_state["reports"] = {}

df = st.session_state.get("df")
summary = st.session_state.get("summary")
quarters = st.session_state.get("quarters", [])
reports = st.session_state.get("reports", {})

if df is None or summary is None:
    st.info("좌측에서 **① 수집/전처리 실행**을 먼저 눌러주세요.")
    st.stop()


# =========================
# Dashboard
# =========================
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("수집 결과(분기별 필터)")
    st.write(f"총 {len(df):,}건 (중복 제거 후)")

    c1, c2, c3 = st.columns(3)
    with c1:
        quarter_sel = st.selectbox("분기 선택", sorted(df["quarter"].unique()))
    with c2:
        sentiment_sel = st.selectbox("감성 선택", ["전체", "긍정", "중립", "부정"], index=0)
    with c3:
        keyword = st.text_input("제목 키워드 필터", value="")

    dff = df[df["quarter"] == quarter_sel].copy()
    if sentiment_sel != "전체":
        dff = dff[dff["sentiment"] == sentiment_sel]
    if keyword.strip():
        dff = dff[dff["title"].str.contains(keyword, case=False, na=False)]

    show_cols = ["published", "sentiment", "title", "source_system", "source", "domain", "url"]
    st.dataframe(dff[show_cols], use_container_width=True, height=420)

    csv_bytes = dff[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="필터 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"KIHS_{quarter_sel}_filtered.csv",
        mime="text/csv",
    )

with right:
    st.subheader("분기별 감성 요약")
    st.dataframe(summary, use_container_width=True, height=200)

    chart_df = summary.set_index("quarter")
    for col in ["긍정", "중립", "부정"]:
        if col not in chart_df.columns:
            chart_df[col] = 0

    st.bar_chart(chart_df[["긍정", "중립", "부정"]], height=240)

    st.markdown("---")
    st.subheader("LLM 분석(Analyze) — Gemini")

    st.caption("분기별로 기사 제목 목록을 요약하고, 정책/기술 시사점을 도출합니다(데모).")

    quarters_list = sorted(df["quarter"].unique())
    gen_target = st.multiselect("분석 대상 분기 선택", options=quarters_list, default=[quarter_sel])

    if analyze_btn:
        new_reports = dict(reports) if isinstance(reports, dict) else {}
        with st.spinner("Gemini로 분석 보고서 생성 중..."):
            for q in gen_target:
                dfq = df[df["quarter"] == q]
                bullets = build_quarter_bullets(dfq, cap=120)
                if not bullets.strip():
                    continue
                try:
                    text = gemini_implications(q, bullets, model_name=model_name)
                    new_reports[q] = text
                except Exception as e:
                    new_reports[q] = f"[오류] Gemini 호출 실패: {e}"

        st.session_state["reports"] = new_reports
        reports = new_reports

    if reports:
        show_q = st.selectbox("보고서 보기", options=sorted(reports.keys()), index=0)
        st.markdown(reports.get(show_q, ""))

        st.download_button(
            label="전체 보고서 JSON 다운로드",
            data=json.dumps(reports, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="KIHS_gemini_reports.json",
            mime="application/json",
        )
    else:
        st.info("아직 생성된 보고서가 없습니다. 좌측에서 수집 후, 우측에서 **② 분석(Analyze) 실행**을 눌러 생성하세요.")

st.markdown("---")
st.caption(
    "주의: 본 앱은 데모용입니다. GDELT는 대량 수집에 유리하지만 출처/언어별 완전한 커버리지를 보장하지 않습니다. "
    "Google News RSS는 보강용이며 포맷/정책/제한이 변동될 수 있습니다."
)
