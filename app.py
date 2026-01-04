import streamlit as st
from google import genai

st.set_page_config(page_title="Gemini API Test", layout="centered")
st.title("Gemini API 연결 테스트")

# 1) Secrets 확인 (키 누락이면 즉시 중단)
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다. (Streamlit Cloud → Settings → Secrets)")
    st.stop()

st.success("✅ GOOGLE_API_KEY 로드 완료")

# 2) Client 생성 (여기서 실패하면 키/패키지/권한 문제)
try:
    client = genai.Client(api_key=api_key)
    st.success("✅ Gemini Client 생성 성공")
except Exception as e:
    st.error("❌ Gemini Client 생성 실패")
    st.exception(e)
    st.stop()

st.markdown("---")

# 3) 간단 입력 + 호출
prompt = st.text_area("프롬프트", value="OK라고만 답해줘.", height=100)

col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("모델", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], index=0)
with col2:
    run = st.button("API 호출 테스트", type="primary")

if run:
    try:
        res = client.models.generate_content(
            model=model,
            contents=prompt
        )
        st.success("🎉 API 호출 성공")
        st.code(res.text or "(빈 응답)")
    except Exception as e:
        st.error("❌ API 호출 실패")
        st.exception(e)
