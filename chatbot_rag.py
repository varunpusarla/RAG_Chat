import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

def process_pdf(pdf_path):
    # Load document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)
    
    # Load embeddings model
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def main():
    st.title("📄 Chat with Your PDF using Ollama & LangChain")
    st.sidebar.header("Upload Your PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        pdf_path = f"temp_{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF
        vectorstore = process_pdf(pdf_path)
        retriever = vectorstore.as_retriever()
        
        # Load Ollama model
        # llm = Ollama(model="llama3.1")
        llm = Ollama(model="llama3.2", base_url="http://localhost:11434")

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        # Chat Interface
        st.subheader("Ask a Question about the PDF")
        user_query = st.text_input("Type your query:")
        if user_query:
            response = qa.run(user_query)
            st.write("**Response:**", response)
        
        # Clean up
        os.remove(pdf_path)

if __name__ == "__main__":
    main()
