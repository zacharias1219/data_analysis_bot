from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from apikey import gemini_api_key

genai.configure(api_key = gemini_api_key)
os.environ['Google_api_key'] = gemini_api_key

def get_model_response(file, query):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)
    context = "\n\n".join(str(p.page_content) for p in file)

    data = text_splitter.split_text(context)
    print(data)

    embeddings = GoogleGenerativeAIEmbeddings(model = "model/embedding-001")
    searcher = Chroma.from_texts(data, embeddings).as_retriever()

    q = "Which employee has maximum salary?"
    records = searcher.get_relevant_documents(q)
    print(records)

    prompt_template = """
            You have to answer the question from the context provided, and make sure that you provide all the details\n
            Context: {context}\n
            Question: {question}\n

            Answer: 
        """
    prompt = PromptTemplate(prompt_template, input_variables = ["context", "question"])

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.9)
    chain = load_qa_chain(model, prompt = prompt, chain_type = "stuff")

    response = chain(
        {
            "input_documents": records,
            "question": query
        }
        , return_only_outputs = True
    )
    return response['output_text']