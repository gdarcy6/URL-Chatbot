import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

def main():
# Add theme configuration at the start of main()
    st.set_page_config(
        page_title="URL Chatbot",
        page_icon=":robot:",
    )
    
    # Enhanced Cyberpunk theme with vibrant text
    st.markdown("""
        <style>
        :root {
            --primary-color: #FE75FE;
        }
        .stApp {
            background-color: #111111;
        }
        .stButton>button {
            background-color: #FE75FE;
            color: #111111;
            font-weight: bold;
            text-shadow: 0 0 2px #FE75FE;
        }
        .stTextInput>div>div>input {
            background-color: #2D1B69;
            color: #FFFFFF;
            font-weight: 500;
        }
        /* Make text more vibrant */
        .stMarkdown {
            color: #FFFFFF !important;
            text-shadow: 0 0 2px rgba(255, 255, 255, 0.3);
        }
        /* Make headers pop with neon effect */
        h1, h2, h3 {
            color: #FFFFFF !important;
            text-shadow: 0 0 10px rgba(254, 117, 254, 0.5),
                         0 0 20px rgba(254, 117, 254, 0.3);
            font-weight: bold !important;
        }
        /* Make regular text more visible */
        p, div {
            color: #FFFFFF !important;
            font-weight: 400;
        }
        /* Style the response text */
        .stMarkdown div[data-testid="stMarkdownContainer"] > p {
            color: #FFFFFF !important;
            font-size: 1.1em;
            line-height: 1.6;
        }
        /* Ensure title is white */
        .css-10trblm {
            color: #FFFFFF !important;
        }
        /* Style text input labels and text */
        .css-1n76uvr {
            color: #FFFFFF !important;
        }
        /* Style response text */
        .stMarkdown div.css-1fttcpj {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("URL Chatbot \N{Robot Face}")
    
    # URL input
    url = st.text_input("Enter the URL you want to analyze:")
    
    if url:
        try:
            # Fetch and parse webpage content
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text content (removing script and style elements)
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            
            # Clean the text
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(chunk for chunk in lines if chunk)
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # Q&A section
            st.subheader("Ask Questions About the Web Page")
            user_question = st.text_input("Ask a question about the content:")
            
            if user_question:
                docs = knowledge_base.similarity_search(user_question)
                
                llm = Ollama(
                    model="llama3",
                    temperature=0.1,
                )
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)
                
                st.write("Answer:")
                st.write(response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
