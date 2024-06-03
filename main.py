import streamlit as st
import langchain_helper as lch
import textwrap

st.title("YouTube Assistant")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.text_area(
            label="What is the YouTube video URL?",
            max_chars=100
        )
        query = st.text_area(
            label="Ask me about the video",
            max_chars=100,
            key="query"
        )

        submit_button = st.form_submit_button(label="Submit")

if submit_button and query and youtube_url:
    db = lch.create_vector_db_from_youtube_url(youtube_url)
    response = lch.get_response_from_query(db, query)
    st.sidebar.write("Answer:")
    st.text(textwrap.fill(response, width=100))