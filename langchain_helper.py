from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful YouTube Assistant that is able to answer questions about the video provided to you based on its transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use factual information from the transcript to answer the question.

        If you feel like you don't have enough information, tell them you do not know.
        """
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.invoke({"question": query, "docs": docs_page_content})

    # Access the 'text' field in the response
    response_text = response['text'].replace("\n", "")
    return response_text
