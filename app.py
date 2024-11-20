import streamlit as st
import openai
import pinecone
import boto3
import json
from sentence_transformers import SentenceTransformer
import pandas as pd

# variables
bucket_name = 'legalprototype'
file_name = 'Cleaned_Contract_Terms.csv'
pinecone_environment = "us-east-1"  # Your Pinecone environment
index_name = "legalprototype"  # Your index name

# keys
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
openai_key = st.secrets["OPENAI_API_KEY"]

# OpenAI API key
client = OpenAI(
  api_key= openai_key,
)

# Validate API keys
if not pinecone_api_key:
    st.error("PINECONE_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()
if not client.api_key:
    st.error("OPENAI_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Initialize S3 client
s3 = boto3.client('s3',
                  aws_access_key_id = aws_access_key_id,
                  aws_secret_access_key = aws_secret_access_key)

# Initialize the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to fetch the document from S3
@st.cache_data
def fetch_document_once():
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=file_name)
        document = pd.read_csv(obj['Body'])
    except Exception as e:
        st.error(f"Error fetching dataset from S3: {str(e)}")
        st.stop()
    return document

# Function to embed a piece of text using OpenAI
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Function to index document chunks in Pinecone
def index_document(document):
    chunk_size = 10  # Adjust chunk size (number of rows per chunk)
    total_rows = len(document)
    
    for i in range(0, total_rows, chunk_size):
        chunk_df = document.iloc[i:i + chunk_size]
        chunk_str = chunk_df.to_string(index=False)
        embedding = get_embedding(chunk_str)
        index.upsert(vectors=[(f"doc_chunk_{i//chunk_size}", embedding, {"text": chunk_str})])

# Cache the document indexing
@st.cache_data
def index_document_once(document):
    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust based on your embedding dimensions
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_environment
            )
        )
    
    # Index the document chunks in Pinecone
    index_document(document)

# Search Pinecone index
def search_pinecone(query_embedding, top_k=1):
    response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return response['matches']

# Function to get contextual data from the document
def get_context(document, question):
    # Try to filter the DataFrame based on relevant keywords in the question
    keywords = question.split()
    filtered_df = document.copy()

    # Filter rows based on keywords that match column values
    for keyword in keywords:
        if keyword in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[keyword].notna()]

    # Limit the number of rows to keep the context manageable
    filtered_df = filtered_df.head(10)

    # Convert the filtered DataFrame to string format
    context = filtered_df.to_string(index=False)
    return context

# Main function to get a response
def get_response(question, document):
    # Step 1: Get context from the document
    context = get_context(document, question)

    # Step 2: Formulate the prompt for OpenAI including the context
    if context.strip():
        messages = [
            {"role": "system", "content": "You are an assistant trained to answer questions based on provided context. If the context is not sufficient, use your general knowledge to answer. In this case, do not say that context was not found."},
            {"role": "user", "content": f"Based on the following context, answer the query: {question}\n\nContext:\n{context}"}
        ]
    else:
        # If no context is found, just use the question to leverage OpenAI's general knowledge
        messages = [
            {"role": "system", "content": "You are an assistant trained to answer questions."},
            {"role": "user", "content": question}
        ]

    # Step 3: Get response from OpenAI
    try:
        openai_response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # gpt-4 if needed
            messages=messages,
            max_tokens=100,
            temperature=0.3
        )
        
        # Extract the reply from the response
        reply = openai_response.choices[0].message['content'].strip()
        return reply
    except Exception as e:
        # Handle exceptions
        return f"Error: {e}"

# Display chat history
def chat_history():
    if 'chat_history' in st.session_state:
        st.write("### Chat History:")
        for message in st.session_state.chat_history:
            st.write(f"**You:** {message['user']}")
            st.write(f"**Bot:** {message['bot']}")

# Example usage
# user question
question = st.text_input("Your question:", key="chat_input")
# Fetch document from S3
document = fetch_document_once()
index_document_once(document)

# Display the answer from model
if st.button("Send"):
    if question:
        try:
            response = get_response(question, document)
            # Append user input and reply to the chat history
            st.session_state.chat_history.append({"user": question, "bot": response})
            # display chat history
            chat_history()
        except openai.error.OpenAIError as e:
            st.error(f"OpenAI API error: {str(e)}")
            st.stop()

