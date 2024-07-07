import streamlit as st
import os
from dotenv import load_dotenv
import fitz #PyMuPDF
from langchain.chains import RetrievalQA
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# model and embedding
emb_model = "models/embedding-001"
llm_model = "gemini-1.5-flash"
CHUNK_SIZE = 720
OVERLAP_SIZE = 150
CHAIN_TYPE = "stuff"

def get_pdf_text(pdf_docs):
    print("LOG: GET PDF TEXT")
    text = ""
    for pdf in pdf_docs:
        pdf_file = fitz.open('documents/'+pdf.name)
        for page_num in range(pdf_file.page_count):
            page = pdf_file.load_page(page_num)
            text += page.get_text()
        pdf_file.close()
    return text


def get_clean_text(text):
    # Remove numbers at the beginning of each line
    text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
    # Remove escape sequences
    text = re.sub(r'\\[^\s]+', '', text)
    # Remove non-English characters and symbols
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text.strip()


def get_text_chunks(text):
    print("LOG: GET TEXT CHUNKS")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    print("LOG: GET VECTOR STORE")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_local")

    
def get_conversational_chain():
    print("LOG: GET CONVERSIONAL CHAIN")

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, 
    if the answer is not in provided context just say, "Answer is not available in the context"\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type=CHAIN_TYPE, prompt=prompt)
    return chain

def get_conversational_chain_2():
    print("LOG: GET CONVERSIONAL CHAIN - 2")

    prompt_template = """
    Firstly go throught the whole text. Answer the question in details, make sure to provide all the details,
    Accordingly format the answer in paragraph and points\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model=llm_model, temperature=0.7)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type=CHAIN_TYPE, prompt=prompt)
    return chain


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role":"assistant",
            "content":"Ask me Anything from the Document"
        }
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def user_input(user_question):
    print("LOG: USER_INPUT()")

    embeddings = GoogleGenerativeAIEmbeddings(model=emb_model)

    new_db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question}
    )
    print(response)

    msg = "Answer is not available in the context"
    if msg in response["output_text"]:

        st.write("Finding the most relevant response....")
        
        chain = get_conversational_chain_2()
        response = chain(
            {"input_documents":docs, "question":user_question}
        )
        print(response)
        # Displaying the Assistant Message
        with st.chat_message("assistant"):
            st.markdown(response["output_text"])

    else:
        with st.chat_message("assistant"):
            st.markdown(response["output_text"])

    # Storing the AI and User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content": user_question
        }
    )
    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": response["output_text"]
        }
    )

# def get_session_messages(data):
#     st.title("Streamlit App: Report Viewer")
    
#     # Loop through each item in the data
#     for item in data:
#         role = item['role']
#         content = item['content']
        
#         # Display based on role
#         if role == 'assistant':
#             st.subheader("Assistant:")
#             st.text(content)
#         elif role == 'user':
#             st.subheader("User:")
#             st.text(content)
#         else:
#             st.subheader("Content:")
#             st.markdown(content)


def main():
    # st.set_page_config("JJM GPT")
    # st.header("Chat with Report")
    # st.write("Powered by Gemini Pro")

    user_question = st.chat_input("What's up?")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        user_input(user_question)


    with st.sidebar:
        st.header("JAL JEEVAN MISSION - INDIA")
        st.title("MENU")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                clean_text = get_clean_text(raw_text)
                text_chunks = get_text_chunks(clean_text)
                get_vector_store(text_chunks)
                st.success("Done")

            with open('raw_text.txt', 'w', encoding='utf-8') as f:
                f.write(raw_text)
            with open('clean_text.txt', 'w', encoding='utf-8') as f:
                f.write(clean_text)
# calling
if __name__=="__main__":
    main()