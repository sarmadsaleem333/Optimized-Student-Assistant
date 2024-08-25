from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

class DocumentQA:
    def __init__(self, text):
        self.document_text = text
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.chunks = self.text_splitter.split_documents(text)
        
        self.embeddings = HuggingFaceEmbeddings()
        self.chunk_embeddings = []
        for chunk in self.chunks:
            embedding = self.embeddings.embed_query(chunk.page_content)
            self.chunk_embeddings.append(embedding)
        self.dimension = self.embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings, dtype='float32'))

    def ask_question(self, question, k=5):
        query_embedding = self.embeddings.embed_query(question)
        query_embedding_array = np.array([query_embedding], dtype='float32')
        _, indices = self.index.search(query_embedding_array, k)
        return [self.chunks[i] for i in indices[0]]
