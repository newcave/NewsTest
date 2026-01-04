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
# - Sources: GDELT + Google News RSS(보강)
# - Quarter UI (분기/브랜치)
# - Gemini 분석 보고서 Pool(누적 저장)
# ============================================================

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="KIHS 온라인 데이터 분석기 (데모)", layout="wide")
st.title("KIHS (한국수자원조사기술원)")
st.caption("온라인 데이터 분석기 (데모) — GDELT + Google News RSS + Gemini")

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
    # { "2024-Q1": {"created_at": "...", "model": "...", "text": "...", "n_items": 123, "query_hash": "..."} }
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
                    "source": a.get("domain"),
                }
            )

        fetched = len(articles)
        startrecord += fetched
        if fetched < params["maxrecords"]:
            break

        time.sleep(0.15)  # 과도한 호출 방지

    return out

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_google_news_rss(query: str, hl="ko", gl="KR", ceid="KR:ko", limit=80):
    # 보강용 RSS
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

# -------------------------
# Normalization / dedup / sentiment (demo rule)
# -------------------------
POS_WORDS = ["확대", "성장", "도입", "개선", "성과", "혁신", "지원", "투자", "상용화", "성공", "협력", "발전"]
NEG_WORDS = ["우려", "논란", "실패", "중단", "규제", "사고", "부족", "지연", "위험", "갈등", "반대"]

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

def query_fingerprint(query: str, start_q: str, end_q: str, use_gdelt: bool, use_rss: bool, gdelt_max: int, rss_max: int):
    s = json.dumps(
        {
            "query": query,
            "start_q": start_q,
            "end_q": end_q,
            "use_gdelt": use_gdelt,
            "use_rss": use_rss,
            "gdelt_max": gdelt_max,
            "rss_max": rss_max,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# -------------------------
# Gemini reporting
# -------------------------
def build_quarter_bullets(dfq: pd.DataFrame, cap: int = 120) -> str:
    d = dfq.copy()
    d["published_dt"] = pd.to_datetime(d["published"], errors="coerce")
    d = d.sort_values("published_dt", ascending=False).drop(columns=["published_dt"])

    lines = []
    for _, r in d.head(cap).iterrows():
        src = r.get("domain") or r.get("source") or r.get("source_system")
        lines.append(f"- [{r['sentiment']}] {r['title']} ({src})")
    return "\n".join(lines)

def gemini_report(quarter: str, bullets: str, model_name: str):
    prompt = f"""
당신은 'KIHS (한국수자원조사기술원) 온라인 데이터 분석기(데모)'의 분석가입니다.

대상 분기: {quarter}

아래 기사/뉴스 제목 목록을 근거로, 과장 없이 간결하고 단정한 톤으로 보고서를 작성하세요.

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

제약:
- 모호한 문장 대신, "무엇을/누가/언제/어떻게"를 포함해 실행형으로 작성.
- 불확실한 경우 '가정'으로 명시.
"""
    res = client.models.generate_content(model=model_name, contents=prompt)
    return res.text or ""

# -------------------------
# Sidebar UI (Branch=Quarter selection)
# -------------------------
with st.sidebar:
    st.header("설정")

    st.subheader("기간(분기/브랜치) 선택")
    c1, c2 = st.columns(2)
    with c1:
        start_q = st.text_input("시작 분기", value="2024-Q1")
    with c2:
        end_q = st.text_input("종료 분기", value="2025-Q1")

    st.subheader("검색어(Query)")
    default_query = (
        '"KIHS" OR "한국수자원조사기술원" OR "Korea Institute of Hydraulic Survey" OR '
        'hydrology OR flood OR drought OR dam OR reservoir OR "water treatment plant" OR wastewater OR leakage OR '
        '"digital twin" OR AI OR "numerical modeling" OR simulation'
    )
    query = st.text_area("검색어", value=default_query, height=140)

    st.subheader("데이터 소스")
    use_gdelt = st.checkbox("GDELT 사용(권장, 대량)", value=True)
    gdelt_max = st.slider("GDELT 분기당 최대 수집", 50, 1000, 250, 50)

    use_rss = st.checkbox("Google News RSS 사용(보강)", value=True)
    rss_max = st.slider("RSS 최대 수집(전체)", 20, 200, 80, 10)

    st.markdown("### ⚠️ RSS(피드) 사용 주의")
    st.warning(
        "RSS는 **보강용**입니다.\n"
        "- 피드 포맷/정책 변경 가능\n"
        "- 커버리지 제한(일부 기사만 노출)\n"
        "- 게시 시각/출처 메타데이터 불완전 가능\n"
        "- 대량 수집/재배포는 약관 이슈 여지\n\n"
        "기간 기반 대량 수집은 **GDELT가 상대적으로 안정적**입니다."
    )

    st.subheader("LLM(Gemini)")
    model_name = st.selectbox("모델", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)

    st.markdown("---")
    btn_collect = st.button("① 수집/전처리", type="primary")
    btn_analyze = st.button("② 분석 보고서 생성", type="secondary")
    btn_clear_pool = st.button("리포트 풀 초기화", type="tertiary")

# -------------------------
# Actions
# -------------------------
if btn_clear_pool:
    st.session_state["report_pool"] = {}
    st.success("리포트 풀을 초기화했습니다.")

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

    # collect GDELT per quarter
    if use_gdelt:
        with st.spinner("GDELT에서 분기별 수집 중..."):
            for qlab, qs, qe in quarters:
                recs = fetch_gdelt_doc(query, qs, qe, max_records=gdelt_max)
                for r in recs:
                    r["quarter"] = qlab
                    all_rows.append(r)

    # collect RSS supplement
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
                r["quarter"] = quarter_label(dt)
                all_rows.append(r)

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
    wanted_quarters = [q for q, _, _ in quarters]
    df = df[df["quarter"].isin(wanted_quarters)].copy()

    # sentiment
    df["sentiment"] = df["title"].apply(rule_sentiment)

    # sort
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.sort_values(["quarter", "published_dt"], ascending=[True, False]).drop(columns=["published_dt"])

    # summary
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

    st.success(f"수집 완료: 총 {len(df):,}건 (중복 제거 후)")

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
    st.info("좌측에서 **① 수집/전처리**를 먼저 실행하세요.")
    st.stop()

# Layout
left, right = st.columns([1.25, 1.0], gap="large")

with left:
    st.subheader("수집 데이터 (분기/감성/키워드 필터)")
    st.write(f"현재 데이터: **{len(df):,}건**")

    f1, f2, f3 = st.columns(3)
    with f1:
        quarter_sel = st.selectbox("분기(브랜치) 선택", sorted(df["quarter"].unique()))
    with f2:
        sentiment_sel = st.selectbox("감성", ["전체", "긍정", "중립", "부정"], index=0)
    with f3:
        kw = st.text_input("제목 키워드", value="")

    dff = df[df["quarter"] == quarter_sel].copy()
    if sentiment_sel != "전체":
        dff = dff[dff["sentiment"] == sentiment_sel]
    if kw.strip():
        dff = dff[dff["title"].str.contains(kw, case=False, na=False)]

    show_cols = ["published", "sentiment", "title", "source_system", "source", "domain", "url"]
    st.dataframe(dff[show_cols], use_container_width=True, height=440)

    csv_bytes = dff[show_cols].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="필터 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"KIHS_{quarter_sel}_filtered.csv",
        mime="text/csv",
    )

with right:
    st.subheader("분기별 감성 요약")
    st.dataframe(summary, use_container_width=True, height=220)

    chart_df = summary.set_index("quarter")
    for col in ["긍정", "중립", "부정"]:
        if col not in chart_df.columns:
            chart_df[col] = 0
    st.bar_chart(chart_df[["긍정", "중립", "부정"]], height=240)

    st.markdown("---")
    st.subheader("분기 분석 보고서 (Report Pool)")
    st.caption("분기별 보고서는 생성 시 '풀'에 저장되며, 재생성/누적 관리가 가능합니다.")

    # pool overview
    if report_pool:
        pool_list = []
        for q, meta in report_pool.items():
            pool_list.append(
                {
                    "분기": q,
                    "생성시각": meta.get("created_at", ""),
                    "모델": meta.get("model", ""),
                    "기사수": meta.get("n_items", 0),
                }
            )
        st.dataframe(pd.DataFrame(pool_list).sort_values("분기"), use_container_width=True, height=160)
    else:
        st.info("아직 생성된 보고서가 없습니다. **② 분석 보고서 생성**을 실행하세요.")

    st.markdown("#### 분석 대상 분기 선택")
    quarters_list = sorted(df["quarter"].unique())
    gen_targets = st.multiselect("보고서 생성(또는 갱신) 분기", quarters_list, default=[quarter_sel])

    # Analyze action
    if btn_analyze:
        fp = query_fingerprint(query, start_q, end_q, use_gdelt, use_rss, gdelt_max, rss_max)
        with st.spinner("Gemini로 보고서 생성 중..."):
            new_pool = dict(report_pool)
            for q in gen_targets:
                dfq = df[df["quarter"] == q]
                bullets = build_quarter_bullets(dfq, cap=120)
                if not bullets.strip():
                    continue
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
                    new_pool[q] = {
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "model": model_name,
                        "n_items": int(len(dfq)),
                        "query_hash": fp,
                        "text": f"[오류] Gemini 호출 실패: {e}",
                    }
            st.session_state["report_pool"] = new_pool
            report_pool = new_pool
        st.success("보고서 생성/갱신이 완료되었습니다.")

    # View report
    st.markdown("---")
    st.markdown("#### 보고서 보기")
    if report_pool:
        view_q = st.selectbox("보고서 선택", options=sorted(report_pool.keys()))
        st.markdown(report_pool[view_q].get("text", ""))

        st.download_button(
            label="리포트 풀 JSON 다운로드",
            data=json.dumps(report_pool, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="KIHS_report_pool.json",
            mime="application/json",
        )
    else:
        st.caption("보고서가 생성되면 이 영역에 표시됩니다.")

st.markdown("---")
st.caption(
    "주의(데모): GDELT는 대량 수집에 유리하나 특정 언론/언어 커버리지를 보장하지 않습니다. "
    "Google News RSS는 보강용이며 포맷/정책/커버리지/메타데이터가 변동될 수 있습니다."
)
