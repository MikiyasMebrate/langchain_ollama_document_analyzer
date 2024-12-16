from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from itertools import islice
import asyncio

# Initialize LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
    stream=True,
)

# Initialize Embeddings
embeddings = OllamaEmbeddings(
    model="znbang/bge:large-en-v1.5-f16",
)

# Load PDF using PyMuPDFLoader
loader = PyMuPDFLoader("./BBIMS.pdf")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Number of characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
)

try:
    # Use lazy_load() to efficiently process large files
    raw_docs = list(islice(loader.lazy_load(), 20))  # Load the first 5 pages lazily

    # Apply text splitting and create LangChain documents
    documents = []
    for raw_doc in raw_docs:
        chunks = text_splitter.split_text(raw_doc.page_content)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=raw_doc.metadata))

except Exception as e:
    print(f"Error loading or processing documents: {e}")
    documents = []

# Assign unique string IDs to documents
document_ids = [str(i) for i in range(len(documents))]

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="foo",
    embedding_function=embeddings,
)

# Add documents to the vector store
try:
    vector_store.add_documents(documents=documents, ids=document_ids)
    print("Documents added successfully!")
except Exception as e:
    print(f"Error adding documents to vector store: {e}")

# Initialize retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up document formatting
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define a prompt template for the chain
template = """
Answer the question based on the context provided, with detailed and specific references only to the context.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)

# Build the chain of operations
chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Async function to handle user interaction
async def ask_questions():
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Stream the response from the chain
        try:
            for chunk in chain.stream(query):  # Pass query as a string
                print(chunk, end='', flush=True)
            print("\n")
        except Exception as e:
            print(f"Error during retrieval or LLM response: {e}")


# Run the async question loop
if __name__ == "__main__":
    asyncio.run(ask_questions())
