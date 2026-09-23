import os
import json
import time
import psycopg2
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SUPABASE_CONN_STRING = os.getenv("SUPABASE_CONN_STRING")

# Ensure the keys are actually found before booting up
if not GOOGLE_API_KEY or not SUPABASE_CONN_STRING:
    raise ValueError("Missing environment variables! Please check your local .env file.")

# Initialize the standard new client
client = genai.Client(api_key=GOOGLE_API_KEY)

print("Loading local embedding model into memory...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Define the structured data schema
class CleanEvent(BaseModel):
    event_name: str = Field(description="The clear title of the volunteer event")
    location: str = Field(description="City, address, or venue of the event")
    event_date: str = Field(description="Date in YYYY-MM-DD format. Use 2026-01-01 if unknown.")
    description: str = Field(description="A clean summary of what volunteers will do")
def call_api_with_retry(api_function, *args, **kwargs):
    """Safely calls Gemini API with exponential backoff if server is busy (503)."""
    max_retries = 5
    delay = 2  # Start with a 2-second delay
    
    for attempt in range(max_retries):
        try:
            return api_function(*args, **kwargs)
        except Exception as e:
            # Check if it's a 503 server error or a temporary Windows socket drop
            if "503" in str(e) or "10054" in str(e):
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️ Gemini server busy or socket dropped. Retrying in {delay} seconds (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2  # Double the wait time for the next try
            else:
                # If it's a code error (like a typo), crash immediately so we can fix it
                raise e
            
def process_file(file_path):
    print(f"Reading file: {file_path}...")
    with open(file_path, 'r') as f:
        raw_text = f.read()

    # Step A: Transform messy text to JSON with explicit AFC disabling
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=f"Extract the event details into structured JSON mapping the schema exactly:\n\n{raw_text}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CleanEvent,
            # Disable automatic tools by passing the explicit config object
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            # Set the configuration mode to the plain string "NONE"
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="NONE"
                )
            )
        ),
    )
    
    clean_data = json.loads(response.text)

    # Step B: Generate text vectors truncated exactly to 768 to match database specifications
    raw_embeddings = embedding_model.encode(clean_data['description'])
    vector_embedding = [float(x) for x in raw_embeddings]


    # Step C: Write to Supabase Postgres
    conn = psycopg2.connect(SUPABASE_CONN_STRING)
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO events (event_name, location, event_date, description, embedding)
    VALUES (%s, %s, %s, %s, %s);
    """
    
    cursor.execute(insert_query, (
        clean_data['event_name'],
        clean_data['location'],
        clean_data['event_date'],
        clean_data['description'],
        vector_embedding
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Success! Saved to Supabase: {clean_data['event_name']}\n")

# Execute the ingestion loop safely
if __name__ == "__main__":
    files_to_process = ["data/file1.txt", "data/file2.txt", "data/file3.txt"]
    
    for file in files_to_process:
        try:
            process_file(file)
            time.sleep(3) # Give the free tier API breathing room between files
        except Exception as e:
            print(f"❌ Failed processing {file}. Error: {e}")
            time.sleep(5)