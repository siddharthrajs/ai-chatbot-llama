import streamlit as st
import fitz 
from langchain.llms import Ollama
import os

def load_pdf(uploaded_file):
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    pdf = fitz.open("uploaded_file.pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def main():
    st.title("PDF Question-Answering Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_text = load_pdf(uploaded_file)

        llm = Ollama(model="llama3") 

        question = st.text_input("Ask a question about the PDF")
        if question:
            prompt = f"Context: {pdf_text}\n\nQuestion: {question}"
            answer = llm.invoke(prompt)
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
