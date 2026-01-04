if st.button("API 연결 테스트"):
    try:
        r = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="OK라고만 답해줘."
        )
        st.success(r.text)
    except Exception as e:
        st.error(str(e))
