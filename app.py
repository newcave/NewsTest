import streamlit as st
from google import genai

# =========================
# App Config
# =========================
st.set_page_config(page_title="Gemini API Test", layout="centered")
st.title("Gemini API 연결 테스트 (Streamlit)")

# =========================
# API Key 확인
# =========================
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
    st.stop()

st.success("✅ GOOGLE_API_KEY 로드 완료")

# =========================
# Gemini Client
# =========================
try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    st.success("✅ Gemini Client 생성 성공")
except Exception as e:
    st.error("❌ Gemini Client 생성 실패")
    st.error(str(e))
    st.stop()

# =========================
# Test Call
# =========================
st.markdown("---")
st.subheader("API 호출 테스트")

if st.button("Gemini API 테스트 실행"):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="OK라고만 답해줘."
        )
        st.success("🎉 API 호출 성공!")
        st.write("Gemini 응답:")
        st.code(response.text)
    except Exception as e:
        st.error("❌ API 호출 실패")
        st.error(str(e))
