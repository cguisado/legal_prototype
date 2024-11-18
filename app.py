import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import pandas as pd
import boto3

# Retrieve API keys from environment variables
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

client = OpenAI(
  api_key=st.secrets["OPENAI_API_KEY"],  # this is also the default, it can be omitted
)
pinecone_environment = "us-east-1"  # Your Pinecone environment
index_name = "legalprototype"  # Your index name

# Validate API keys
if not pinecone_api_key:
    st.error("PINECONE_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()
if not client.api_key:
    st.error("OPENAI_API_KEY is not set. Please configure it in the Streamlit Cloud environment variables.")
    st.stop()

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

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

index = pc.Index(index_name)

# Load the dataset
# Replace 'your_access_key_id' and 'your_secret_access_key' with your actual AWS credentials
s3 = boto3.client('s3',
                  aws_access_key_id = aws_access_key_id,
                  aws_secret_access_key = aws_secret_access_key)

bucket_name = 'legalprototype'
file_name = 'Cleaned_Contract_Terms.csv'

# Read the CSV file directly from S3
try:
    obj = s3.get_object(Bucket=bucket_name, Key=file_name)
    df = pd.read_csv(obj['Body'])
except Exception as e:
    st.error(f"Error fetching dataset from S3: {str(e)}")
    st.stop()

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

        # Extract relevant column descriptions from Pinecone results
        column_context = "\n".join([f"{match['metadata']['description']}" for match in response.matches])

        # Extract relevant column names for filtering the DataFrame
        relevant_columns = [match['metadata']['column_name'] for match in response.matches if 'column_name' in match['metadata']]

        # Filter the DataFrame based on relevant columns
        if relevant_columns:
            filtered_df = df[relevant_columns]
        else:
            filtered_df = pd.DataFrame()

        # Construct context from filtered DataFrame
        if not filtered_df.empty:
            context = ""
            for _, row in filtered_df.head(10).iterrows():  # Limit rows for token optimization
                context += f"{row.to_string(index=False)}\n"
        else:
            context = "No relevant data found for the query."

        # Combine context with column descriptions
        full_context = f"Column Descriptions:\n{column_context}\n\nRelevant Data:\n{context}"

        # Generate a response
        try:
            openai_response = client.chat.completions.create(
                model="gpt-4",  # Use "gpt-3.5-turbo" if GPT-4 is not available
                messages=[
                    {"role": "system", "content": "You are an assistant trained to answer questions based on provided context."},
                    {"role": "user", "content": f"Based on the following context, answer the query: {user_input}\n\nContext:\n{full_context}"}
                ],
                max_tokens=150,
                temperature=0.7
            )

            # Extract the reply from the response
            reply = openai_response.choices[0].message.content.strip()

            # Append user input and reply to the chat history
            st.session_state.chat_history.append({"user": user_input, "bot": reply})
          
        except client.error.OpenAIError as e:
            st.error(f"OpenAI API error: {str(e)}")
            st.stop()

# Display chat history
st.write("### Chat History:")
for message in st.session_state.chat_history:
    st.write(f"**You:** {message['user']}")
    st.write(f"**Bot:** {message['bot']}")
