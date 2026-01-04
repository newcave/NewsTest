import streamlit as st
from google import genai

st.set_page_config(page_title="Gemini API Test", layout="centered")
st.title("Gemini API 연결 테스트")

# 1) 키 존재 확인
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다. (Streamlit Cloud → Settings → Secrets)")
    st.stop()

st.success("✅ GOOGLE_API_KEY 로드 완료")

# 2) 클라이언트 생성
try:
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    st.success("✅ Gemini Client 생성 성공")
except Exception as e:
    st.error("❌ Gemini Client 생성 실패")
    st.exception(e)
    st.stop()

# 3) 버튼은 반드시 import/설정 이후에
if st.button("API 연결 테스트"):
    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="OK라고만 답해줘."
        )
        st.success("🎉 API 호출 성공")
        st.write(r.text)
    except Exception as e:
        st.error("❌ API 호출 실패")
        st.exception(e)
