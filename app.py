import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage

ollama_model = ChatOllama(model="llama3", server="http://localhost:11434")

if ollama_model.model == "llama3.2:1b":
    model_name = "Llama3.2 1B"
elif ollama_model.model == "llama3.2":
    model_name = "Llama3.2 3B"
elif ollama_model.model == "llama3.1":
    model_name = "Llama3.1 8B"
elif ollama_model.model == "llama3":
    model_name = "Llama3"
else:
    model_name = "Unknown model"

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


def get_conversation_chain(vectorstore):
    llm = ollama()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_response(human_message, text_chunks):
    
    prompt = f"""
    ### Document Processing and Question

    **Provided Document/Text:**
    {text_chunks}

    **User's Question:**
    {human_message}?

    **Expected Response:**
    Please provide a detailed and accurate response considering the document content and the user's question.
    """
    human_message = HumanMessage(content=prompt)
    response = ollama_model([human_message], temperature=0.2, max_tokens=512)
    return response

def main():
    # this is the ui of the page
    st.set_page_config(page_title="Chat with mutliple PDFs", page_icon="📖")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDFs :books:")
    human_message = st.text_input("Ask a question about your documents:")
    # human_message.split('\n')[-1]
    

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
                st.session_state.text_chunks = get_text_chunks(raw_text)
                st.write(st.session_state.text_chunks)
                st.write(human_message.split('\n')[-1])

                # convert to embeddings
                # embeddings = get_embeddings(text_chunks)
                # st.write(embeddings)

                # create vector store
                # vectorstore = get_vectorstore(text_chunks)
                # st.write(vectorstore)

                # get question from the user

                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(vectorstore)
                # st.write(st.session_state.conversation)

    if human_message:
        # human_message = HumanMessage(content=human_message)
        response = get_response(human_message, st.session_state.text_chunks)
        st.write(response.content)





if __name__ == "__main__":
    main()