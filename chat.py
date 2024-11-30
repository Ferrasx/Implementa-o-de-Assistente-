import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "multilingual-e5-large"
# Configurar a chave de API do OpenAI
openai.api_key = OPENAI_API_KEY
st.write(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}")

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Funções para extrair texto de PDF e gerar embeddings
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_embeddings(text):
    sentences = text.split(". ")
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings

def store_embeddings(embeddings, sentences):
    vectors = [(str(i), embeddings[i].tolist(), {"text": sentences[i]}) for i in range(len(sentences))]

def query_assistant(question):

    # Gerar a resposta usando o OpenAI
    context = " ".join([match["metadata"]["text"] for match in results["matches"]])
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um assistente inteligente."},
            {"role": "user", "content": f"Contexto: {context}\n\nPergunta: {question}"}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"]

# Interface com Streamlit
st.write("Carregue um arquivo PDF e faça perguntas baseadas no conteúdo!")

uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processando o arquivo PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        sentences, embeddings = generate_embeddings(text)
        store_embeddings(embeddings, sentences)

    st.write("Agora, você pode fazer perguntas sobre o conteúdo do arquivo.")
    question = st.text_input("Digite sua pergunta:")

    if question:
        with st.spinner("Gerando resposta..."):
            answer = query_assistant(question)
            st.write("Resposta:", answer)
