import logging
from openai import OpenAI
import pdfplumber
import requests
from bs4 import BeautifulSoup
import tiktoken
import os
from decouple import Config as DecoupleConfig, RepositoryEnv
import time
from docx import Document
import ollama
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Load the LLaMA model
desiredModel='Hudson/llama3.1-uncensored:8b'
questionToAsk = """You are to clean up the following text. You do not provide any input, explainations, introductions, analysis, metadata, nothing at all. You retain the text as is, but you format and provide spelling, grammar and general correction to errors. Do as follows:
        1. Format the paragraphs correctly.
        2. Remove unnecessary line breaks and repetitions.
        3. Focus on improving readability while preserving all original content.

        IMPORTANT: I reiterate Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.
        """

# set flag to decide which model to use
USE_OPENAI = False  

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.StreamHandler(),
    logging.FileHandler("processing.log")  
])

config = DecoupleConfig(RepositoryEnv('.env'))

OPENAI_API_KEY = config.get("OPENAI_API_KEY", default="your-openai-api-key", cast=str)


api_key = OPENAI_API_KEY
if not api_key:
    logging.error("API key not found. Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key) 

def extract_text_from_pdf(pdf_path):
    logging.info(f"Extracting text from PDF: {pdf_path}")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        logging.warning(f"Failed to extract text from PDF: {e}")
    return text

def extract_text_from_url(url):
    logging.info(f"Extracting text from URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        logging.warning(f"Error fetching URL {url}: {e}")
        return ""

def extract_text_from_file(file_path):
    logging.info(f"Extracting text from file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.warning(f"Failed to extract text from file: {e}")
        return ""

# split Text into chunks for large inputs
def split_text(text, max_tokens=100): 
    logging.info("Splitting text into chunks while preserving sentence boundaries.")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    # split the text into sentences
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        sentence_token_count = len(sentence_tokens)
        
        # if adding the current sentence would exceed the max_tokens limit, start a new chunk
        if current_token_count + sentence_token_count > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_token_count = sentence_token_count
        else:
            current_chunk += " " + sentence
            current_token_count += sentence_token_count

    # append the last chunk if there's any leftover text
    if current_chunk:
        chunks.append(current_chunk)

    logging.info(f"Total chunks created: {len(chunks)}")

    with open('chunks_before_processing.txt', 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"Chunk {i}:\n{chunk}\n{'-'*50}\n")

    return chunks


def analyze_text_with_openai(text, prompt):
    logging.info("Analyzing text using OpenAI API.")
    try:
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            stream=True,
        )
        response_text = ""
        for chunk in stream:
            content = getattr(chunk.choices[0].delta, 'content', '')
            if content is not None:
                response_text += content
        return response_text.strip()
    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}")
        return ""

def analyze_text_locally(text, prompt):
    logging.info("Analyzing text using LLaMA model locally.")
    try:
        # Generate response using LLaMA
        output = ollama.chat(model=desiredModel, messages=[
            {
                'role': 'system',
                'content': prompt
            },
            {
                'role': 'user',
                'content': text
            }
        ])
        
  
        logging.info(f"Ollama response: {output}")

        if isinstance(output, dict) and 'message' in output and 'content' in output['message']:
            OllamaResponse = output['message']['content']
            return OllamaResponse.strip()
        else:
            logging.error("Unexpected response format from Ollama.")
            return ""
        
    except Exception as e:
        logging.error(f"Error during LLaMA model inference: {e}")
        return ""

def analyze_text(text, prompt):
    if USE_OPENAI:
        return analyze_text_with_openai(text, prompt)
    else:
        return analyze_text_locally(text, prompt)

def create_analysis_report(text):
    logging.info("Creating a analysis report by splitting text into chunks.")
    prompt = (
        """Analyze the following text and clean it up making it readable:
        1. Format the paragraphs correctly.
            -   Paragraph endings must make sense.
            -   Remove unnecessary line breaks within sentences or paragraphs
            -   Sometimes in older documents, the text repeats itself at the end of paragraphs. Please pay attention to it and remove it
        2. Any text that represents the chapter title, book title, or page number that you can infer remove totally.
        3. Do not CHANGE any of the content. Words are to remain the same as the author intended. Everything you read will remain the same as is. The goal is to structure the document, not alter the content.

        IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.

        """
    )
    return analyze_text(text, prompt)

def create_large_analysis_report(text):
    logging.info("Creating a large analysis report by splitting text into chunks.")
    folder_name = input("Enter the name of the folder to save chunks: ").strip()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    chunks = split_text(text)
    full_report = ""
    for i, chunk in enumerate(chunks, start=1):
        logging.info(f"Processing chunk {i}/{len(chunks)}.")
        corrected_text = None
        retries = 3 
        for attempt in range(retries):
            corrected_text = create_analysis_report(chunk)

            if corrected_text:
                break
            else:
                logging.warning(f"Retrying chunk {i}. Attempt {attempt + 1} of {retries}.")
        
        if corrected_text:
            chunk_output_path = os.path.join(folder_name, f"chunk_{i}_{int(time.time())}.txt")
            with open(chunk_output_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(corrected_text + "")
            logging.info(f"Chunk {i} saved to {chunk_output_path}.")
            full_report += corrected_text + "\n"
        else:
            logging.error(f"Chunk {i} could not be processed after {retries} attempts and was skipped.")
    return full_report

def save_report_to_text(report, folder_name, output_path=None):
    logging.info(f"Saving report to text file in folder: {folder_name}")
    try:
        if output_path is None:
            output_path = os.path.join(folder_name, f"generated_report_{int(time.time())}.txt")
        logging.info(f"Saving report to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(report + "")
        logging.info(f"Report successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save report to text file: {e}")

def generate_report(input_list):
    logging.info("Generating report from input list.")
    full_text = ""
    for input_type, input_value in input_list:
        if input_type == 'pdf':
            full_text += extract_text_from_pdf(input_value) + "\n"
        elif input_type == 'url':
            full_text += extract_text_from_url(input_value) + "\n"
        elif input_type == 'file':
            full_text += extract_text_from_file(input_value) + "\n"
        else:
            logging.warning(f"Invalid input type: {input_type}")
            raise ValueError("Invalid input type. Use 'pdf', 'url', or 'file'.")


    if len(full_text) > 3000: 
        logging.info("Text is too large for a single request. Creating a large analysis report.")
        report = create_large_analysis_report(full_text)
    else:
        report = create_analysis_report(full_text)
    

    folder_name = input("Enter the name of the folder to save the final report: ").strip()
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_report_to_text(report, folder_name)
    return report

if __name__ == "__main__":
    input_list = []
    while True:
        input_type = input("Enter input type (pdf, url, file) or 'done' to finish: ").strip().lower()
        if input_type == 'done':
            break
        elif input_type in ['pdf', 'url', 'file']:
            input_value = input("Enter the path to the file or URL: ").strip()
            input_list.append((input_type, input_value))
        else:
            logging.warning("Invalid input type entered.")
            print("Invalid input type. Please enter 'pdf', 'url', 'file', or 'done'.")

    try:
        report = generate_report(input_list)
        logging.info("Report generation completed successfully.")
        print("\nGenerated Report:\n")
        if report:
            print(report)
        if report:
            save_report_to_text(report)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
