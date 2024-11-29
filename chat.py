
import openai
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import os
# Configuração das chaves usando variáveis de ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
INDEX_NAME = "multilingual-e5-large"

# Configurar a chave de API do OpenAI
openai.api_key = OPENAI_API_KEY
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

# Funções para trabalhar com PDF e embeddings
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
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
    
    # Realizar a consulta ao Pinecone usando argumentos nomeados
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

# Função principal
def main():
    print("=== Assistente Conversacional ===")
    pdf_path = input("Insira o caminho do arquivo PDF: ").strip()
    text = extract_text_from_pdf(pdf_path)

    print("Processando o arquivo PDF...")
    sentences, embeddings = generate_embeddings(text)
    store_embeddings(embeddings, sentences)
    print("Arquivo processado com sucesso e embeddings armazenados no Pinecone!")

    while True:
        question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ").strip()
        if question.lower() == "sair":
            break
        answer = query_assistant(question)
        print(f"Resposta: {answer}")

# Executa o programa
if __name__ == "__main__":
    main()
