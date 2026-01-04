import streamlit as st

st.set_page_config(page_title="Test", layout="centered")
st.title("Streamlit 동작 테스트")

# st는 import 이후에만 사용 가능
if st.button("API 연결 테스트"):
    st.success("버튼 클릭 OK")

