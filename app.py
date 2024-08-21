import streamlit as st
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from cryptography.fernet import Fernet
import io
import zipfile
import os
from cryptography.fernet import Fernet
import os
import logging
import time
import csv
import re
from typing import Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cryptography.fernet import Fernet



# Create a sidebar with tabs
tab = st.sidebar.radio("Select a page:", ["Chatbot", "Download Chat History","Decruption","Generate Dataset"])

if tab == "Chatbot":
    st.title("ChatCinema")

    # Step 1: Read the CSV file
    csv_file_path = 'Hydra-Movie-Scrape.csv'
    df = pd.read_csv(csv_file_path)

    # Ensure there are no NaN values in the 'Summary' column
    df['Summary'] = df['Summary'].fillna('')

    # Use 'Summary' for embeddings
    summaries = df['Summary'].tolist()

    # Step 2: Generate embeddings for the data
    model = SentenceTransformer('all-MiniLM-L6-v2')
    summary_embeddings = model.encode(summaries)

    # Step 3: Store embeddings in a vector database
    vector_database = np.array(summary_embeddings)

    # Step 4: Define a function to retrieve the most similar entry from the vector database
    def retrieve_similar_movie(query, vector_db, top_k=1):
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, vector_db)
        top_indices = np.argsort(similarities[0])[::-1][:top_k]
        return df.iloc[top_indices[0]], top_indices

    secrets = st.secrets["groq_api_key"]

# Use the API key to create a Groq client
    client = Groq(api_key=secrets)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(f"*You:* {message['content']}")
            else:
                st.write(message["content"])

    # React to user input
    user_query = st.chat_input("Enter Movie Name...")

    if user_query:
        # Add the user's input prompt to the chat history
        st.session_state.messages.append({"role": "user", "content": f"**You:** {user_query}"})

        # Retrieve the most similar movie and its details from the vector database
        retrieved_movie, _ = retrieve_similar_movie(user_query, vector_database)

        # Prepare the message content using the retrieved data
        retrieved_content = f"Context: {retrieved_movie['Summary']}"
        answer_title = retrieved_movie['Title']

        # Use the retrieved content as context for the chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question, but feel free to expand or provide additional information if necessary."},
                {"role": "user", "content": retrieved_content},
                {"role": "user", "content": user_query},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        model_output = chat_completion.choices[0].message.content

        # Display the output
        with st.chat_message("assistant"):
          st.write(f"**Title:** {retrieved_movie['Title']}")
          st.write(f"**Year:** {retrieved_movie['Year']}")
          st.write(f"**Summary:** {retrieved_movie['Summary']}")
          st.write(f"**Genres:** {retrieved_movie['Genres']}")
          st.write(f"**IMDB ID:** {retrieved_movie['IMDB ID']}")
          st.write(f"**YouTube Trailer:** {retrieved_movie['YouTube Trailer']}")
          st.write(f"**Rating:** {retrieved_movie['Rating']}")
          st.write(f"**Movie Poster:** {retrieved_movie['Movie Poster']}")
          st.write(f"**Director:** {retrieved_movie['Director']}")
          st.write(f"**Writers:** {retrieved_movie['Writers']}")
          st.write(f"**Cast:** {retrieved_movie['Cast']}")
          st.write(f"**Dialogues:** {model_output}")

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": model_output})



                
        new_data = {
            'Prompt': [user_query],
            'Title': [retrieved_movie['Title']],
            'Year': [retrieved_movie['Year']],
            'Summary': [retrieved_movie['Summary']],
            'Genres': [retrieved_movie['Genres']],
            'IMDB ID': [retrieved_movie['IMDB ID']],
            'YouTube Trailer': [retrieved_movie['YouTube Trailer']],
            'Rating': [retrieved_movie['Rating']],
            'Movie Poster': [retrieved_movie['Movie Poster']],
            'Director': [retrieved_movie['Director']],
            'Writers': [retrieved_movie['Writers']],
            'Cast': [retrieved_movie['Cast']],
            'Detailed Explanation': [model_output]
        }

        # Create DataFrame for the new data
        new_df = pd.DataFrame(new_data)

        # Check if the file exists
        output_csv_path = 'summary_output.csv'
        if os.path.exists(output_csv_path):
            # If it exists, append to the existing file
            existing_df = pd.read_csv(output_csv_path)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If it does not exist, just use the new DataFrame
            updated_df = new_df

        # Save the updated DataFrame back to the CSV file
        updated_df.to_csv(output_csv_path, index=False)

elif tab == "Download Chat History":
    st.header("Download your History")
    csv_file_path = 'summary_output.csv'
    df = pd.read_csv(csv_file_path)
    if st.button("Download History"):
        
        # Generate a key for encryption
        key = Fernet.generate_key()
        cipher_suite = Fernet(key)

        # Convert DataFrame to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')

        # Encrypt the CSV data
        encrypted_data = cipher_suite.encrypt(csv_data)

        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr('encrypted_data.csv', encrypted_data)
            zf.writestr('encryption_key.key', key)

        # Ensure the zip file is ready for download
        zip_buffer.seek(0)

        # Download button for the zip file containing the encrypted CSV and the key
        st.download_button(
            label="Download Encrypted CSV and Key",
            data=zip_buffer,
            file_name='encrypted_data_and_key.zip',
            mime='application/zip'
        )


elif tab == "Decruption":


  # Function to decrypt data using the provided key
  def decrypt_data(encrypted_data, key):
      fernet = Fernet(key)
      return fernet.decrypt(encrypted_data)

  # Function to decrypt CSV file
  def decrypt_csv(uploaded_file, key):
      encrypted_data = uploaded_file.read()
      decrypted_data = decrypt_data(encrypted_data, key)
      return decrypted_data.decode()

  # Streamlit UI
  st.title('CSV File and Text Decryption')

  # Upload CSV file
  uploaded_file = st.file_uploader("Upload Encrypted CSV File", type=["csv"])

  # Input key for decryption
  key = st.text_input("Enter the decryption key", type="password")

  # Decrypt button
  if st.button("Decrypt CSV File"):
      if uploaded_file is not None and key:
          try:
              decrypted_csv_content = decrypt_csv(uploaded_file, key.encode())
              st.success("CSV file decrypted successfully!")
              
              # Convert decrypted content to a DataFrame for download
              decrypted_csv_df = pd.read_csv(io.StringIO(decrypted_csv_content))
              
              # Convert the DataFrame back to CSV for download
              decrypted_csv_data = decrypted_csv_df.to_csv(index=False).encode('utf-8')
              
              # Provide a download button for the decrypted CSV file
              st.download_button(label="Download Decrypted CSV",
                                data=decrypted_csv_data,
                                file_name="decrypted_file.csv",
                                mime="text/csv")
          except Exception as e:
              st.error(f"An error occurred: {str(e)}")
      else:
          st.warning("Please upload a file and enter the decryption key.")

  # Decrypt an encrypted text
  st.write("### Decrypt Encrypted Text")
  encrypted_text = st.text_area("Enter the encrypted text")
  text_decrypt_key = st.text_input("Enter the key for text decryption", type="password")

  if st.button("Decrypt Text"):
      if encrypted_text and text_decrypt_key:
          try:
              decrypted_text = decrypt_data(encrypted_text.encode(), text_decrypt_key.encode())
              st.success("Text decrypted successfully!")
              st.write("Decrypted Text:")
              st.write(decrypted_text.decode())
          except Exception as e:
              st.error(f"An error occurred: {str(e)}")
      else:
          st.warning("Please provide both the encrypted text and the key.")



if tab == "Generate Dataset":



  # Function to generate model output
  def generate(
      prompt: str,
      model: Union[str, AutoModelForCausalLM],
      hf_access_token: str,
      tokenizer: Union[str, AutoTokenizer] = 'meta-llama/Llama-2-7b-hf',
      device: Optional[str] = None,
      max_length: int = 1024,
      assistant_model: Optional[Union[str, AutoModelForCausalLM]] = None,
      generate_kwargs: Optional[dict] = None,
  ) -> str:
      """Generates output given a prompt."""
      if not device:
          if torch.cuda.is_available() and torch.cuda.device_count():
              device = "cuda:0"
          else:
              device = 'cpu'

      if isinstance(model, str):
          model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
      model.to(device).eval()

      if isinstance(tokenizer, str):
          tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=hf_access_token)

      # Prepare the prompt
      tokenized_prompt = tokenizer(prompt)
      tokenized_prompt = torch.tensor(tokenized_prompt['input_ids'], device=device)
      tokenized_prompt = tokenized_prompt.unsqueeze(0)

      # Generate
      output_ids = model.generate(
          tokenized_prompt,
          max_length=max_length,
          pad_token_id=tokenizer.pad_token_id,
          **(generate_kwargs if generate_kwargs else {}),
      )

      output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

      return output_text

  def extract_qa(text):
      """Extract question and answer pairs from generated text."""
      qas = []
      parts = text.split("\n")  # Split by line breaks

      for i in range(0, len(parts) - 1, 2):
          question = parts[i].strip()
          answer = parts[i + 1].strip()
          if question and answer:
              qas.append((question, answer))

      return qas

  # Streamlit UI
  st.title('Encrypted Dataset Generator')

  # Input fields for two prompts
  prompt1 = st.text_input("Enter the first prompt:")
  prompt2 = st.text_input("Enter the second prompt:")

  if st.button("Generate Dataset"):
      if prompt1 and prompt2:
          # Prepare encryption key
          key = Fernet.generate_key()
          cipher_suite = Fernet(key)

          # Prepare CSV file (in-memory storage)
          output_stream = io.StringIO()
          csvwriter = csv.writer(output_stream)
          csvwriter.writerow(['Generated Question', 'Generated Answer'])

          # Define input arguments for the generation function
          args = {
              'model': 'apple/OpenELM-270M',
              'hf_access_token': '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',
              'device': None,
              'max_length': 350,
              'assistant_model': None,
              'generate_kwargs': {'repetition_penalty': 2.0},
          }

          # Generate and process prompts
          for prompt in [prompt1, prompt2]:
              output_text = generate(
                  prompt=prompt,
                  model=args['model'],
                  device=args['device'],
                  max_length=args['max_length'],
                  assistant_model=args['assistant_model'],
                  generate_kwargs=args['generate_kwargs'],
                  hf_access_token=args['hf_access_token'],
              )

              qas = extract_qa(output_text)
              for question, answer in qas:
                  csvwriter.writerow([question, answer])

          # Encrypt the CSV data
          csv_data = output_stream.getvalue().encode('utf-8')
          encrypted_data = cipher_suite.encrypt(csv_data)
          # Create a ZIP file in memory
          zip_buffer = io.BytesIO()
          with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
              # Add encrypted CSV data to the ZIP
              zip_file.writestr('encrypted_ai_qa_dataset.csv', encrypted_data)
              # Add encryption key to the ZIP
              zip_file.writestr('encryption_key.key', key)

          # Set the buffer's position to the beginning
          zip_buffer.seek(0)

          # Provide download link for the ZIP file
          st.download_button(
              label="Download ZIP FILE",
              data=zip_buffer,
              file_name="Encrypted_Dataset.zip",
              mime="application/zip"
          )

      else:
          st.warning("Please enter both prompts.")
