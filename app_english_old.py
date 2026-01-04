import re
import json
import time
import hashlib
from datetime import datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import quote

import pandas as pd
import requests
import feedparser
import streamlit as st
from google import genai

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="KIHS Online Data Analyzer (Demo)",
    layout="wide",
)

st.title("Korea Institute of Water Resources Research (KIHS)")
st.caption("Monthly Related Online Data Analyzer (Demo Version) — GDELT + Google News RSS + Gemini")

# =========================
# Secrets / API
# =========================
# Streamlit Cloud Secrets:
# GOOGLE_API_KEY = "AIza..."
API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
if not API_KEY:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. Streamlit Cloud → Settings → Secrets에 GOOGLE_API_KEY를 추가하세요.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# =========================
# Utilities: Quarter
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

        # polite delay (avoid hammering)
        time.sleep(0.2)

    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_google_news_rss(query: str, hl: str = "ko", gl: str = "KR", ceid: str = "KR:ko", limit: int = 100):
    """
    Google News RSS (supplement)
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
# Normalization / Dedup / Sentiment
# =========================
POS_WORDS = ["확대", "성장", "도입", "개선", "성과", "혁신", "지원", "투자", "상용화", "성공", "협력", "발전"]
NEG_WORDS = ["우려", "논란", "실패", "중단", "규제", "사고", "부족", "지연", "위험", "갈등", "반대", "피해"]


def rule_sentiment(text: str) -> str:
    t = (text or "").lower()
    p = sum(w in t for w in POS_WORDS)
    n = sum(w in t for w in NEG_WORDS)
    if p > n:
        return "positive"
    if n > p:
        return "negative"
    return "neutral"


def make_key(url: str, title: str) -> str:
    base = (url or "").strip()
    if not base:
        base = (title or "").strip()
    return hashlib.md5(base.encode("utf-8", errors="ignore")).hexdigest()


def within_range(published_iso: str, start_dt: datetime, end_dt: datetime) -> bool:
    if not published_iso:
        return False
    try:
        dt = datetime.fromisoformat(published_iso)
        return start_dt <= dt < end_dt
    except Exception:
        return False


# =========================
# Gemini: Implications
# =========================
def gemini_implications(quarter: str, bullets: str, model_name: str, temperature: float, max_output_tokens: int):
    prompt = f"""
You are an analyst for:
"Korea Institute of Water Resources Research (KIHS)
Monthly Related Online Data Analyzer (Demo Version)".

Quarter: {quarter}

News bullets:
{bullets}

Write a structured report in Korean with the following sections:
1) 분기 요약 (6줄 이내)
2) 긍정 요인 (bullet)
3) 부정 요인 (bullet)
4) 향후 정책 시사점 (3~6개 bullet, 실행가능하게)
5) 향후 기술 시사점 (3~6개 bullet, 실행가능하게)
6) 다음 분기 모니터링 키워드 (10개)
7) 전제/리스크 (bullet)

Be concise, direct, and formal.
"""
    # google-genai: generation config can be passed via config=...
    # Keep it simple for compatibility.
    res = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return res.text


# =========================
# Sidebar UI
# =========================
with st.sidebar:
    st.header("Settings")

    colA, colB = st.columns(2)
    with colA:
        start_q = st.text_input("Start Quarter", value="2024-Q1")
    with colB:
        end_q = st.text_input("End Quarter", value="2025-Q1")

    default_query = (
        '"Korea Institute of Water Resources Research" OR KIHS OR '
        'hydrology OR flood OR drought OR dam OR reservoir OR "water treatment plant" OR wastewater OR leakage OR '
        '"digital twin" OR AI OR "numerical modeling" OR simulation'
    )
    query = st.text_area("Search Query (GDELT/RSS)", value=default_query, height=120)

    st.subheader("Sources")
    use_gdelt = st.checkbox("Use GDELT (bulk)", value=True)
    use_rss = st.checkbox("Use Google News RSS (supplement)", value=True)

    gdelt_max = st.slider("GDELT max per quarter", min_value=50, max_value=1000, value=250, step=50)
    rss_max = st.slider("RSS max total", min_value=20, max_value=200, value=80, step=10)

    st.subheader("Gemini")
    model_name = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)
    gen_temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_output_tokens = st.slider("Max output tokens (approx)", 256, 4096, 900, 128)

    run_btn = st.button("Run Collection / Analysis", type="primary")


# =========================
# Main Logic
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


def run_pipeline(start_q: str, end_q: str, query: str, use_gdelt: bool, use_rss: bool, gdelt_max: int, rss_max: int):
    # Validate quarters
    try:
        # Iterate to validate
        quarters = list(quarter_iter(start_q, end_q))
        if not quarters:
            raise ValueError("empty")
    except Exception:
        st.error("Quarter 형식이 올바르지 않습니다. 예: 2024-Q1")
        return None, None

    all_rows = []

    # Collect per quarter from GDELT
    if use_gdelt:
        with st.spinner("Collecting from GDELT by quarter..."):
            for qlabel, qs, qe in quarters:
                recs = fetch_gdelt_doc(query, qs, qe, max_records=gdelt_max)
                for r in recs:
                    r["quarter"] = qlabel
                    r["source"] = r.get("domain")
                    all_rows.append(r)

    # Collect RSS (not perfectly filterable)
    if use_rss:
        with st.spinner("Collecting from Google News RSS (supplement)..."):
            rss_recs = fetch_google_news_rss(query, limit=rss_max)
            # Assign quarter only if in the selected range
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
                # keep only those within requested quarters
                all_rows.append(r)

    if not all_rows:
        st.warning("수집 결과가 없습니다. Query 또는 기간/소스를 조정해 주세요.")
        return None, None

    df = pd.DataFrame(all_rows)

    # Normalize essential columns
    for c in ["title", "url", "published", "source_system", "quarter", "domain", "language", "snippet", "source"]:
        if c not in df.columns:
            df[c] = None

    df["key"] = [make_key(u, t) for u, t in zip(df["url"].astype(str), df["title"].astype(str))]
    df = df.drop_duplicates(subset=["key"]).copy()

    # Filter quarters strictly within range labels
    wanted_quarters = [q for q, _, _ in quarters]
    df = df[df["quarter"].isin(wanted_quarters)].copy()

    # Sentiment
    df["sentiment"] = df["title"].apply(rule_sentiment)

    # Sort
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.sort_values(["quarter", "published_dt"], ascending=[True, False]).drop(columns=["published_dt"])

    # Summaries
    summary = (
        df.groupby(["quarter", "sentiment"])
        .size()
        .reset_index(name="count")
        .pivot(index="quarter", columns="sentiment", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    return df, summary


if run_btn:
    df, summary = run_pipeline(start_q, end_q, query, use_gdelt, use_rss, gdelt_max, rss_max)
    st.session_state["df"] = df
    st.session_state["summary"] = summary
    st.session_state["reports"] = {}  # reset reports on new run

# Show results if available
df = st.session_state.get("df")
summary = st.session_state.get("summary")
reports = st.session_state.get("reports", {})

if df is None or summary is None:
    st.info("좌측에서 설정 후 **Run Collection / Analysis**를 눌러 시작하세요.")
    st.stop()

# =========================
# Dashboard
# =========================
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.subheader("Collected Articles (Filtered by Quarter)")
    st.write(f"총 {len(df):,}건 (중복 제거 후)")

    # Filtering UI
    c1, c2, c3 = st.columns(3)
    with c1:
        quarter_sel = st.selectbox("Quarter", sorted(df["quarter"].unique()))
    with c2:
        sentiment_sel = st.selectbox("Sentiment", ["all", "positive", "neutral", "negative"], index=0)
    with c3:
        keyword = st.text_input("Title keyword filter", value="")

    dff = df[df["quarter"] == quarter_sel].copy()
    if sentiment_sel != "all":
        dff = dff[dff["sentiment"] == sentiment_sel]
    if keyword.strip():
        dff = dff[dff["title"].str.contains(keyword, case=False, na=False)]

    show_cols = ["published", "sentiment", "title", "source_system", "source", "domain", "url"]
    st.dataframe(dff[show_cols], use_container_width=True, height=420)

    # Downloads
    csv_bytes = dff[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="Download filtered CSV",
        data=csv_bytes,
        file_name=f"KIHS_{quarter_sel}_filtered.csv",
        mime="text/csv",
    )

with right:
    st.subheader("Quarter Sentiment Summary")
    st.dataframe(summary, use_container_width=True, height=200)

    # Simple chart (no manual colors)
    chart_df = summary.set_index("quarter")
    # Ensure columns exist
    for col in ["positive", "neutral", "negative"]:
        if col not in chart_df.columns:
            chart_df[col] = 0
    st.bar_chart(chart_df[["positive", "neutral", "negative"]], height=240)

    st.markdown("---")
    st.subheader("LLM Implications (Gemini)")

    # Choose which quarters to generate
    quarters_list = sorted(df["quarter"].unique())
    gen_target = st.multiselect("Generate reports for quarters", options=quarters_list, default=[quarter_sel])

    if st.button("Generate / Update Reports"):
        new_reports = dict(reports) if isinstance(reports, dict) else {}
        with st.spinner("Generating reports with Gemini..."):
            for q in gen_target:
                dfq = df[df["quarter"] == q]
                bullets = build_quarter_bullets(dfq, cap=120)
                if not bullets.strip():
                    continue
                try:
                    text = gemini_implications(
                        quarter=q,
                        bullets=bullets,
                        model_name=model_name,
                        temperature=gen_temperature,
                        max_output_tokens=max_output_tokens,
                    )
                    new_reports[q] = text
                except Exception as e:
                    new_reports[q] = f"[ERROR] Gemini call failed: {e}"
        st.session_state["reports"] = new_reports
        reports = new_reports

    if reports:
        # show selected quarter report first
        show_q = st.selectbox("View report", options=sorted(reports.keys()), index=0)
        st.markdown(reports.get(show_q, ""))

        st.download_button(
            label="Download all reports (JSON)",
            data=json.dumps(reports, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="KIHS_gemini_reports.json",
            mime="application/json",
        )
    else:
        st.info("아직 생성된 리포트가 없습니다. 'Generate / Update Reports'를 눌러 생성하세요.")

st.markdown("---")
st.caption(
    "Notes: GDELT는 전세계 뉴스 기반으로 대량 수집이 가능하나, 특정 출처/언론사/언어의 완전한 커버리지는 보장되지 않습니다. "
    "RSS는 보강용이며 포맷/제한이 변동될 수 있습니다."
)
