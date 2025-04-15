import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-001",temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.markdown(f"<p style='font-family: Quantico; font-size: 16px;'>Reply: {response['output_text']}</p>", unsafe_allow_html=True)

def main():
    st.set_page_config("EoD Chatbot", layout="wide")

    # Inject custom CSS
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quantico:ital,wght@0,400;0,700;1,400;1,700&display=swap');

    body, .stApp {
        font-family: 'Quantico';
        background-image: url('/static/blank-dark-wall-living-room.jpg');

    }

    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.18);
        color: #ffffff;
    }

    .eod-header h1 {
        font-family: "Quantico";
        color: #FFD700;
    }

    .stTextInput > div > div > input {
        border-radius: 100px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        color: #ffffff;
        font-family: 'Quantico';
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eod-header">', unsafe_allow_html=True)
    st.markdown("<h1 style='font-family: Quantico, sans-serif;'>End of Day (EoD) Report Summarizer</h1>", unsafe_allow_html=True)

    #st.header("End of Day (EoD) Report Summarizer")
    st.markdown('</div>', unsafe_allow_html=True)

    user_question = st.text_input("", placeholder="How can I help?")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload EoD files and Click on the Initialize Button", accept_multiple_files=True)
        if pdf_docs:
            total_size = sum(file.size for file in pdf_docs)
            st.metric(label="Total size", value=f"{total_size / (1024 * 1024):.2f} MB")
        st.subheader("_Developed by_ :rainbow[**Teja Prabhu**]")
        if st.button("Initialize"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Render Success")

if __name__ == "__main__":
    main()
