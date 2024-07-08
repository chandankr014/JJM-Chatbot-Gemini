import streamlit as st
import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Variables
emb_model = "models/embedding-001"
llm_model = "gemini-1.5-flash"
CHUNK_SIZE = 720
OVERLAP_SIZE = 150
CHAIN_TYPE = "stuff"

# Translator    
translator = Translator()

def translate_text(text, target_language):
    if target_language == 'en':
        return text
    translation = translator.translate(text=text, dest=target_language)
    return translation.text

def get_conversational_chain():
    print("LOG: GET CONVERSATIONAL CHAIN")
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
    print("LOG: GET CONVERSATIONAL CHAIN - 2")
    prompt_template = """
    Firstly go through the whole text. Answer the question in details, make sure to provide all the details,
    Accordingly format the answer in paragraph and points\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model=llm_model, temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type=CHAIN_TYPE, prompt=prompt)
    return chain

# Initialize chat history and logs
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me Anything"
        }
    ]

if "language" not in st.session_state:
    st.session_state.language = "en"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# LOAD JSON QnA File
with open("suggested_questions.json", 'r') as file:
    qna_list = json.load(file)

# Function to suggest similar questions
def suggest_similar_questions(user_question, qna_list, top_n=4):
    questions = [qna["question"] for qna in qna_list]
    vectorizer = TfidfVectorizer().fit_transform(questions + [user_question])
    vectors = vectorizer.toarray()
    
    user_vector = vectors[-1]
    question_vectors = vectors[:-1]
    
    cosine_similarities = cosine_similarity([user_vector], question_vectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    suggestions = [questions[i] for i in related_docs_indices]
    return suggestions

def get_answer_for_question(qna_list, que):
    for item in qna_list:
        if item.get("question") == que:
            return item.get("answer")
    return "Question not found."

# Function to perform YouTube search
def youtube_search(query: str, language: str):
    api_key = os.getenv('YOUTUBE_API_KEY')
    search_url = "https://www.googleapis.com/youtube/v3/search"
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    
    translated_query = query + " in Jal jeevan mission"
    
    search_params = {
        'part': 'snippet',
        'q': translated_query,
        'key': api_key,
        'maxResults': 5,
        'type': 'video'
    }
    
    try:
        search_response = requests.get(search_url, params=search_params)
        search_results = search_response.json()

        video_ids = [item['id']['videoId'] for item in search_results['items']]
        
        video_params = {
            'part': 'snippet,statistics',
            'id': ','.join(video_ids),
            'key': api_key
        }
        
        video_response = requests.get(video_url, params=video_params)
        video_results = video_response.json()
        
        videos = []
        for item in video_results.get('items', []):
            title = translate_text(item['snippet']['title'], language)
            video_data = {
                'title': title,
                'url': f"https://www.youtube.com/watch?v={item['id']}",
            }
            videos.append(video_data)
        
        return videos
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []


def user_input(user_question, language):
    print("LOG: USER_INPUT()")
    embeddings = GoogleGenerativeAIEmbeddings(model=emb_model)
    new_db = FAISS.load_local("faiss_local", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})

    msg = "answer is not available in the context"
    if msg in response["output_text"].lower():
        # st.write("Finding the most relevant response....")
        chain = get_conversational_chain_2()
        response = chain({"input_documents": docs, "question": user_question})
        output_text = translate_text(response["output_text"], language)
        with st.chat_message("assistant"):
            st.markdown(output_text)

        # Suggest FAQ questions
        suggestions = suggest_similar_questions(user_question, qna_list)
        st.session_state.suggested_questions = suggestions
    else:
        output_text = translate_text(response["output_text"], language)
        with st.chat_message("assistant"):
            st.markdown(output_text)

    # # Storing the User Message
    # st.session_state.messages.append({"role": "user", "content": user_question})
    # # Storing the Assistant Message
    # st.session_state.messages.append({"role": "assistant", "content": output_text})

def main():
    # Sidebar for language selection and search type
    st.sidebar.header("Settings")
    st.sidebar.title("Jal Jeevan Mission Chatbot")
    search_type = st.sidebar.radio(
        "Search Type:",
        ("Chatbot", "YouTube")
    )
    language = st.sidebar.selectbox(
        "Select Language",
        ("English", "Hindi", "Kannada" , "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Malayalam", "Punjabi")
    )
    language_code = {
        "English": "en", "Hindi": "hi", "Kannada": "kn", "Tamil": "ta", "Telugu": "te", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu", "Malayalam": "ml", "Punjabi": "pa"
    }
    
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []

    if search_type == "Chatbot":
        user_question = st.text_input("Enter your query for Chatbot")
        if user_question:
            st.info(f"Processing query for chatbot: '{user_question}'")
            user_input(user_question, language_code[language])

        # FAQ
        if st.session_state.suggested_questions:
            st.write("Here are some relevant questions you might find helpful:")
            if language=="English" or language=="en":
                selected_question = st.selectbox("Select Most Relevant Question", st.session_state.suggested_questions)
                if selected_question:
                    answer = get_answer_for_question(qna_list, selected_question)
                    st.markdown(answer)
                    
            else:
                # create session 
                st.session_state.sq_translate = []
                suggested_questions_dict = {}
                for que in st.session_state.suggested_questions:
                    sugg_que = translate_text(que, language)
                    ans = get_answer_for_question(qna_list, que)
                    sugg_ans = translate_text(ans, language)
                    suggested_questions_dict[sugg_que] = sugg_ans
                    st.session_state.sq_translate.append(sugg_que)
                selected_question = st.selectbox("Select Most Relevant Question", st.session_state.sq_translate)
                answer = suggested_questions_dict[selected_question]
                st.markdown(answer)
                
    elif search_type == "YouTube":
        user_question = st.text_input("Enter your query for Youtube")
        if user_question:
            st.info(f"Searching YouTube for: '{user_question}'")
            youtube_videos = youtube_search(user_question, language_code[language])
            if youtube_videos:
                st.subheader("Search Results:")
                for video in youtube_videos:
                    st.markdown(f"[{video['title']}]({video['url']})")
            else:
                st.warning("No videos found.")
        else:
            st.warning("Please enter a query for YouTube.")

    # Sidebar footer
    st.sidebar.markdown("\n\n\n\n\n\n\n-------")
    st.sidebar.markdown("© JJM - IIM Bangalore Cell 2024")
    st.sidebar.markdown("-------")

if __name__ == "__main__":
    main()
