import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import openai
import os

# Retrieve API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_environment = "us-east-1"  # Your Pinecone environment
index_name = "legalprototype"  # Your index name

# Validate API keys
if not pinecone_api_key:
    st.error("PINECONE_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()
if not openai.api_key:
    st.error("OPENAI_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Streamlit UI
st.title("Chat with Your Data")
st.write("Type your question below and receive answers based on your indexed embeddings.")

# Initialize the embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize or retrieve chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input for the chat
user_input = st.text_input("Your question:", key="chat_input")
if st.button("Send"):
    if user_input:
        # Generate embedding for the user query
        query_vector = embedding_model.encode(user_input).tolist()

        # Query Pinecone
        response = index.query(
            namespace="solar_contracts",  # Use the namespace used during upsert
            vector=query_vector,
            top_k=3,  # Retrieve the top 3 matches
            include_values=True,
            include_metadata=True
        )

        # Construct context from Pinecone results
        context = "\n".join([f"{match['metadata']['description']}" for match in response.matches])

        openai_response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if GPT-4 is not available
            messages=[
                {"role": "system", "content": "You are an assistant trained to answer questions based on provided context."},
                {"role": "user", "content": f"Based on the following context, answer the query: {user_input}\n\nContext:\n{context}\n\nAnswer:"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        reply = openai_response["choices"][0]["message"]["content"].strip()

        # Append user input and reply to the chat history
        st.session_state.chat_history.append({"user": user_input, "bot": reply})

# Display chat history
st.write("### Chat History:")
for message in st.session_state.chat_history:
    st.write(f"**You:** {message['user']}")
    st.write(f"**Bot:** {message['bot']}")