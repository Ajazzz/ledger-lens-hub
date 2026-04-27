import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone

# Load environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorStore")

class LedgerLensIndexer:
    def __init__(self, data_dir: str = "structured_data"):
        self.data_dir = Path(data_dir).resolve()
        
        #Embeddings
        
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Ensure index exists in Pinecone Dashboard (Dimension 384 for bge-small)
        self.pinecone_index = self.pc.Index(self.index_name)

    def run_indexing(self):
        logger.info("📑 Loading Markdown data...")
        #
        documents = SimpleDirectoryReader(input_dir=str(self.data_dir)).load_data()

        #
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} semantic nodes from 10-K.")

        # 4. Connect to Pinecone and Upload
        vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("📤 Uploading vectors to Pinecone...")
        self.index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context,
            show_progress=True
        )
        logger.info("✅ Success! Pinecone index is now live.")

if __name__ == "__main__":
    indexer = LedgerLensIndexer()
    indexer.run_indexing()
