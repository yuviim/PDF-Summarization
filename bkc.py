from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # FAISS should come from langchain.vectorstores
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI  # Correct path for ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from api_key import api_key
import os

def process_text(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    knowledgebase=FAISS.from_texts(chunks,embeddings)
    
    return knowledgebase

def summarizer(pdf):
    if pdf is not None:
        os.environ["OPENAI_API_KEY"] = api_key

        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        knowledgeBase = process_text(text)

        query = 'Summarize the content of the uploaded pdf file in approximately 10 sentences'

        if query:
            docs = knowledgeBase.similarity_search(query)

            OpenAIModel = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)

            chain = load_qa_chain(llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query)

            return response

    # In case pdf is None or something goes wrong
    return "No PDF uploaded or processing failed."
