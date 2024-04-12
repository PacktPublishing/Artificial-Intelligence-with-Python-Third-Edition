import langchain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Initialize the LLM
my_key = "<your api key here>"
llm = OpenAI(openai_api_key=my_key)

# Load the text file
loader = TextLoader('./sample_text.txt')
documents = loader.load()

# Chunk the text into smaller pieces
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Initialize embeddings model and in-memory vector store
embeddings_model = OpenAIEmbeddings(openai_api_key=my_key)
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings_model)

# Setup the retriever
retriever = vectorstore.as_retriever()

# Create a prompt template
template = """
You are an assistant trained to provide answers based on specific text.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the chat model and the RAG chain

rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

# Example query
query = "What are the main themes discussed in the text?"
answer = rag_chain.invoke(query)
print(answer)
