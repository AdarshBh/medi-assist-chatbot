import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain.chains import create_retrieval_chain
from langchain.chat_models import ChatCohere
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def build_chain(text_chunks):
    # Init embeddings + Pinecone
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # if not pc.has_index(index_name):
    #     pc.create_index(
    #         name=index_name,
    #         dimension=384,
    #         metric="cosine",
    #         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    #     )
    # index = pc.Index(index_name)
    
    # text_vector = PineconeVectorStore.from_documents(
    #     documents=text_chunks,
    #     index_name=index_name,
    #     embedding=embeddings
    # )

    vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Initializing LLM
    llm = ChatCohere(
        model="command-r-plus",
        temperature=0.3,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    # Prompt logic
    short_prompt = (
        "You are a concise medical assistant. If you don't know the answer, say" 
        "that out of medical field so you don't know Answer the question in 1-2 "
        "sentences. Keep it simple and medically accurate.\n\n"
        "{context}"
    )
    long_prompt = (
        "You are a helpful and detailed medical assistant. Use the context to answer the question in "
        "5-7 informative sentences. Avoid numbered lists unless explicitly requested. Write naturally, like "
        "a doctor explaining to a patient. If you don't know the answer, say that out of medical field so "
        "you don't know. \n\n"
        "{context}"
    )

    def dynamic_prompt(query):
        system = short_prompt if len(query.split()) < 6 else long_prompt
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}")
        ])

    def get_chain(query):
        prompt = dynamic_prompt(query)
        qa_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, qa_chain)

    return get_chain
