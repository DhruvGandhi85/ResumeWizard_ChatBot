import streamlit as st
import store_in_redis
import search_in_redis
import os

embedding_model = "nomic-embed-text"
chat_model = "mistral"

st.title("Resume Wizard")

uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

if uploaded_file is not None:
    try:
        # Save the uploaded file temporarily
        uploaded_file_path = "temp_resume.pdf"  # Define a temporary file name
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"File '{uploaded_file.name}' uploaded successfully.")

        # Process the uploaded PDF
        with st.spinner("Processing your resume..."):
            resume_text_chunks = store_in_redis.convert_pdf_to_text(uploaded_file_path)
            full_resume_text = " ".join(resume_text_chunks)

        st.subheader("Your Uploaded Resume:")
        st.text_area("", full_resume_text, height=300)

        # Get feedback on the resume
        with st.spinner("Getting resume improvement suggestions..."):
            default_query = 'Please elaborate on how the resume can be improved, and what should be added or removed based on similar entries.'
            input_query = st.text_input("Enter a specific question about your resume:", value=default_query)
            if input_query:
                feedback = search_in_redis.interactive_search(
                    embedding_model=embedding_model,
                    chat_model=chat_model,
                    query=input_query,
                    breakpoint=True
                )
        st.subheader("Resume Improvement Suggestions:")
        if feedback:
            st.markdown(feedback)
        else:
            st.info("No immediate feedback received.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)

else:
    st.info("Please upload your PDF resume to get improvement suggestions.")