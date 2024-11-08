import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # initialize pdf reader object
        pdf_reader = PdfReader(pdf)

        # loop through each page in pdf
        for page in pdf_reader.pages:
            # concatenate text 
            text += page.extract_text()

    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    embed = OllamaEmbeddings(
        model="llama3:latest"
    )
    embeddings = embed.embed_documents(text_chunks)
    return embeddings

def get_vectorstore(text_chunks):
    embed = OllamaEmbeddings(
        model="llama3:latest"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embed)
    return vectorstore

# def get_conversation_chain(vectorstore):
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )
#     conversation_chain = 

def main():
    # this is the ui of the page
    st.set_page_config(page_title="Chat with mutliple PDFs", page_icon="📖")

    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.header("Insight Ledger")
        st.subheader("Your documents")

        # take pdf as input
        pdf_docs = st.file_uploader("Upload your pdfs here and click on process", 
                                    accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # extract text from pdfs
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # divide into text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # convert to embeddings
                # embeddings = get_embeddings(text_chunks)
                # st.write(embeddings)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.write(vectorstore)

                # get question from the user

                # create conversation chain
                # conversation = get_conversation_chain(vectorstore)






if __name__ == "__main__":
    main()