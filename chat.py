import openai
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# Configuração das chaves
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_77Mnm8_NQVjdoykkLdzPkuGmcbfN1MJ2kUK3u64iZMyxGVwFpvCEwkg7K6u17pGFfqRRXD"
PINECONE_ENVIRONMENT = "us-east1-gcp"
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


# Funções para trabalhar com PDF e embeddings
#def extract_text_from_pdf(pdf_path):
    #reader = PdfReader(pdf_path)
    #text = ""
    #for page in reader.pages:
#   text += page.extract_text()
    #return text


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


# Código da aplicação com Streamlit
st.title("Assistente Conversacional com Pinecone e OpenAI")
st.write("Envie um arquivo PDF, armazene os dados no Pinecone e faça perguntas sobre o conteúdo.")

# Carregar o arquivo PDF
uploaded_file = st.file_uploader("Carregar PDF", type="pdf")
if uploaded_file is not None:
    # Extrair texto do PDF
    text = extract_text_from_pdf(uploaded_file)

    if text:
        # Gerar e armazenar embeddings
        sentences, embeddings = generate_embeddings(text)
        store_embeddings(embeddings, sentences)
        st.success("Arquivo processado com sucesso e embeddings armazenados no Pinecone!")
    else:
        st.error("Não foi possível extrair texto do arquivo PDF.")

# Pergunta ao assistente
question = st.text_input("Digite sua pergunta sobre o conteúdo do PDF:")
if st.button("Perguntar"):
    if question:
        answer = query_assistant(question)
        st.write(f"Resposta: {answer}")
    else:
        st.error("Por favor, insira uma pergunta válida.")


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
