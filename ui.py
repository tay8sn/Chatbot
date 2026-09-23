import os
import streamlit as st
import psycopg2
from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

SUPABASE_CONN_STRING = os.getenv("SUPABASE_CONN_STRING")

# Ensure the keys are actually found before booting up
if not SUPABASE_CONN_STRING:
    raise ValueError("Missing environment variables! Please check your local .env file.")

@st.cache_resource
def load_model():  # <-- We will explicitly name it load_model here
    return SentenceTransformer("all-mpnet-base-v2")

embedding_model = load_model()

st.title("🌱 NGO Volunteer AI Coordinator")
st.write("Ask the AI chatbot to find relevant local community events!")

user_query = st.text_input("How would you like to help today?", placeholder="e.g., I want to help with animals or nature...")

if user_query:
    with st.spinner("Searching matching events..."):
        try:
            # 1. Generate clean list vector
            raw_embeddings = embedding_model.encode(user_query)
            query_vector = [float(x) for x in raw_embeddings]

            # 2. Run explicit type-casted PostgreSQL vector scan
            conn = psycopg2.connect(SUPABASE_CONN_STRING)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT event_name, location, event_date, description FROM match_events(%s::vector, 0.2, 3);", 
                (query_vector,)
            )
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()

            # 3. Dynamic display
            if results:
                st.subheader("🎯 Recommended Matches for You:")
                for row in results:
                    with st.expander(f"📌 {row[0]} — {row[1]}"):
                        st.write(f"**Date:** {row[2]}")
                        st.write(f"**Details:** {row[3]}")
            else:
                st.warning("No matching events found. Try a different phrase!")
                
        except Exception as e:
            st.error(f"An error occurred during search: {e}")
