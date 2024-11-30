import openai
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = "multilingual-e5-large"

# Configurar a chave de API do OpenAI
openai.api_key = OPENAI_API_KEY

# Inicialize o Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Verifique se o índice já existe, caso contrário, crie-o
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Dimensão correta para o modelo "all-MiniLM-L6-v2"
        metric="cosine",  # Métrica de similaridade
        spec=ServerlessSpec(
            cloud="gcp",
            region=PINECONE_ENVIRONMENT
        )
    )

# Conecte-se ao índice usando o Pinecone Index
pinecone_index = pc.Index(INDEX_NAME)

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
    pinecone_index.upsert(vectors)

def query_assistant(question):
    # Gerar o embedding da pergunta
    query_embedding = embedding_model.encode([question])[0]

    # Realizar a consulta ao Pinecone
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=3,
        include_values=False,
        include_metadata=True
    )

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
st.title("Assistente Conversacional com Pinecone e OpenAI")
st.write("Carregue um arquivo PDF e faça perguntas baseadas no conteúdo!")

uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processando o arquivo PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        sentences, embeddings = generate_embeddings(text)
        store_embeddings(embeddings, sentences)
        st.success("Arquivo processado com sucesso e embeddings armazenados no Pinecone!")

    st.write("Agora, você pode fazer perguntas sobre o conteúdo do arquivo.")
    question = st.text_input("Digite sua pergunta:")

    if question:
        with st.spinner("Gerando resposta..."):
            answer = query_assistant(question)
            st.write("Resposta:", answer)
