import streamlit as st 
import os

from bkc import *

from api_key import api_key


def main():
    st.title("PDF Summarizer using LLMs")
    
    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_pdf is not None:
        with st.spinner("Processing the PDF..."):
            response = summarizer(uploaded_pdf)
        st.success("Summary generated!")
        st.button("Generate Summary")
        st.write(response)
    else:
        st.info("Please upload a PDF to get started.")

if __name__ == "__main__":
    main()
    