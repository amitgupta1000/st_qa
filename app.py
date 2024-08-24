__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import RetrievalQA
from langchain.schema import Document
import numpy as np
import nltk
import logging

google_api_key = st.secrets["GOOGLE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

os.environ["GOOGLE_API_KEY"] == st.secrets["GOOGLE_API_KEY"]
os.environ["GROQ_API_KEY"] == st.secrets["GROQ_API_KEY"]

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load text from different file types
def load_text(file_path: str, file_type: str) -> str:
    try:
        if file_type == "pdf":
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        elif file_type == "text":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_type == "excel":
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
        elif file_type == "csv":
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        return text
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

# Function to combine sentences for context
def combine_sentences(sentences: List[Dict[str, Any]], buffer_size: int = 1) -> List[Dict[str, Any]]:
    for i in range(len(sentences)):
        combined_sentence = ''
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '
        combined_sentence += sentences[i]['sentence']
        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']
        sentences[i]['combined_sentence'] = combined_sentence.strip()
    return sentences

# Function to calculate cosine distances between sentences
def calculate_cosine_distances(sentences: List[Dict[str, Any]]) -> (List[float], List[Dict[str, Any]]):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance
    return distances, sentences

# Streamlit app
def main():
    st.title("Stateful Question-Answering System")
    
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'all_text' not in st.session_state:
        st.session_state.all_text = ""
    if 'questions_asked' not in st.session_state:
        st.session_state.questions_asked = []
    if 'answers' not in st.session_state:
        st.session_state.answers = []

    # File upload section
    file_type = st.selectbox("Select file type:", ["pdf", "excel", "csv", "text", "done"])
    
    if file_type == "done":
        if not st.session_state.uploaded_files:
            st.error("No content loaded from the specified files. Exiting.")
            return
    else:
        file_path = st.text_input("Enter the file path:", key="file_path")
        if st.button("Load File"):
            if file_path:
                text = load_text(file_path, file_type)
                if text is not None:
                    st.session_state.all_text += text + "\n"  # Combine text from all files
                    st.session_state.uploaded_files.append((file_type, file_path))
                    st.success(f"Loaded content from {file_path}")
                else:
                    st.error(f"Failed to load content from {file_path}.")
            else:
                st.error("Please enter a valid file path.")

    # Process the loaded text only if there is any
    if st.session_state.all_text:
        # Tokenize sentences
        sentences = nltk.sent_tokenize(st.session_state.all_text)
        sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]

        # Combine sentences for context
        sentences = combine_sentences(sentences)

        # Generate embeddings
        googlegenai_embeds = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
        embeddings = googlegenai_embeds.embed_documents([x['combined_sentence'] for x in sentences])

        # Add embeddings to sentences
        for i, sentence in enumerate(sentences):
            sentence['combined_sentence_embedding'] = embeddings[i]

        # Calculate cosine distances
        distances, sentences = calculate_cosine_distances(sentences)

        # Determine breakpoints and create chunks
        breakpoint_distance_threshold = np.percentile(distances, 95)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

        chunks = []
        start_index = 0
        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)
            start_index = index + 1
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        documents = [Document(page_content=chunk) for chunk in chunks]
        llm = ChatGroq(model='llama-3.1-70b-versatile', groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.7)
        vectorstore = Chroma.from_documents(documents=documents, embedding=googlegenai_embeds)

        # Create a QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Question-Answering Section
        user_input = st.text_input("Ask your question:")
        
        if st.button("Submit"):
            if user_input:
                # Remember the question
                st.session_state.questions_asked.append(user_input)

                # Extract the answer text from the result dictionary
                result = qa_chain.invoke(user_input)
                answer = result['result']
                
                # Store the answer
                st.session_state.answers.append(answer)

                # Display the answer
                st.write("Assistant:")
                st.write(answer)

    # End of session display
    if st.button("End Session"):
        if st.session_state.questions_asked:
            st.write("### Questions and Answers:")
            for question, answer in zip(st.session_state.questions_asked, st.session_state.answers):
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {answer}")

            # Prepare to save to a text file
            questions_answers = "\n".join(
                [f"Q: {q}\nA: {a}" for q, a in zip(st.session_state.questions_asked, st.session_state.answers)]
            )
            st.download_button("Download Q&A", questions_answers, file_name="qa_session.txt")

if __name__ == "__main__":
    main()
