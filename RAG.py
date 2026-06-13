import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# CHANGE 1: Import the native Memory Store instead of Chroma
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Setup API Key 
os.environ["GOOGLE_API_KEY"] = "APIkey" # <-- Don't forget to paste your key here!

# 2. Load the PDF 
loader = PyPDFLoader(r"yourFile.pdf") 
pages = loader.load()

# 3. Split the text 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages)

# 4. Create the Vector Database 
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# CHANGE 2: Build the database using InMemoryVectorStore
vector_db = InMemoryVectorStore.from_documents(chunks, embeddings)

# 5. Setup Gemini 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

# 6. Ask the Question!
question = "According to this document, what is the core topic?"
response = rag_chain.invoke(question)

print("\n--- AI Answer ---")
print(response["result"])
