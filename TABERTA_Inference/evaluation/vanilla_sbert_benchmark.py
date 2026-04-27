
#for the beast

import qdrant_client
import hashlib
import time
from typing import List, Set
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vanilla_embedding_store.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-mpnet-base-v2"
QDRANT_COLLECTION = "benchmark_vanilla_sbert"
BATCH_SIZE = 50
TIMEOUT = 60

class QdrantManager:
    def __init__(self):
        self.client = qdrant_client.QdrantClient(
            "localhost",
            port=6333,
            timeout=TIMEOUT,
            prefer_grpc=False
        )
        self.existing_ids = set()

    def initialize_collection(self, model):
        """Initialize collection and cache existing IDs"""
        if not self.client.collection_exists(QDRANT_COLLECTION):
            logger.info(f"🚀 Creating new collection: {QDRANT_COLLECTION}")
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )
        else:
            logger.info(f"🔍 Loading existing IDs from {QDRANT_COLLECTION}")
            self._cache_existing_ids()

    def _cache_existing_ids(self):
        """Cache all existing point IDs to prevent duplicates"""
        try:
            scroll_result = self.client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100000,
                with_payload=False,
                with_vectors=False
            )
            self.existing_ids = {point.id for point in scroll_result[0]}
            logger.info(f"📦 Loaded {len(self.existing_ids)} existing embeddings")
        except Exception as e:
            logger.error(f"Failed to cache existing IDs: {str(e)}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
    def safe_upsert(self, points: List[PointStruct]):
        """Store embeddings with duplicate check"""
        new_points = [p for p in points if p.id not in self.existing_ids]
        
        if not new_points:
            return

        try:
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=new_points,
                wait=True
            )
            # Update cache with new IDs
            self.existing_ids.update(p.id for p in new_points)
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            time.sleep(5)
            raise

def generate_table_id(db_name: str, table_name: str, table_text: str) -> int:
    """Generate unique ID using content-based hashing"""
    unique_string = f"{db_name}::{table_name}::{table_text}"
    return int(hashlib.md5(unique_string.encode()).hexdigest()[:16], 16)

def process_table(wikidb, db_name: str, table_name: str, model) -> PointStruct:
    """Process a table only if it doesn't exist in Qdrant"""
    try:
        if table_name not in wikidb.list_collection_names():
            logger.warning(f"Skipping {table_name}: Table not found")
            return None

        # Generate table content
        table_data = list(wikidb[table_name].find({}, {"_id": 0}).limit(100))
        table_text = " ".join(f"{k}: {v}" for doc in table_data[:3] for k, v in doc.items())
        table_id = generate_table_id(db_name, table_name, table_text)

        # Create embedding only if new
        embedding = model.encode(table_text, show_progress_bar=False)
        
        return PointStruct(
            id=table_id,
            vector=embedding.tolist(),
            payload={
                "database": db_name,
                "table": table_name,
                "text": table_text[:1000]  # Store first 1000 characters
            }
        )
    except Exception as e:
        logger.error(f"❌ Failed processing {table_name}: {str(e)}")
        return None

def store_vanilla_embeddings():
    """Main function with duplicate prevention"""
    qdrant_mgr = QdrantManager()
    model = SentenceTransformer(MODEL_NAME).to("cpu")
    
    try:
        # Initialize collection and ID cache
        qdrant_mgr.initialize_collection(model)

        with MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=15000) as mongo_client:
            wikidb = mongo_client["wikidbs_10k"]
            BASE_PATH = "/datastore/servers/Enas/enas/tabert/wikidbs-10k/databases"
            
            for db_name in os.listdir(BASE_PATH):
                batch = []
                tables = [col for col in wikidb.list_collection_names() if col.startswith(db_name)]
                
                for table_name in tables:
                    point = process_table(wikidb, db_name, table_name, model)
                    if point:
                        batch.append(point)
                        
                        if len(batch) >= BATCH_SIZE:
                            qdrant_mgr.safe_upsert(batch)
                            batch = []
                            time.sleep(1)

                if batch:
                    qdrant_mgr.safe_upsert(batch)

    except Exception as global_error:
        logger.error(f"💥 Critical failure: {str(global_error)}")
        raise

if __name__ == "__main__":
    logger.info("🚀 Starting embedding storage with duplicate prevention")
    start_time = time.time()
    store_vanilla_embeddings()
    duration = time.time() - start_time
    logger.info(f"✅ Completed in {duration//3600:.0f}h {(duration%3600)//60:.0f}m {duration%60:.0f}s")


# import qdrant_client
# import hashlib
# import time
# from typing import List
# from qdrant_client.models import VectorParams, Distance, PointStruct
# from sentence_transformers import SentenceTransformer
# from pymongo import MongoClient
# from tenacity import retry, stop_after_attempt, wait_exponential
# import logging
# import os

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("vanilla_embedding_store.log"), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# MODEL_NAME = "all-mpnet-base-v2"
# QDRANT_COLLECTION = "benchmark_vanilla_sbert"
# BATCH_SIZE = 50  # Reduced batch size for stability
# TIMEOUT = 60  # Increased timeout to 60 seconds

# class QdrantManager:
#     def __init__(self):
#         self.client = qdrant_client.QdrantClient(
#             "localhost",
#             port=6333,
#             timeout=TIMEOUT,
#             prefer_grpc=False  # Disable gRPC to prevent connection issues
#         )

#     def recreate_collection(self):
#         """Delete the collection if it exists, then recreate it."""
#         if self.client.collection_exists(QDRANT_COLLECTION):
#             logger.info(f"🗑 Deleting existing collection: {QDRANT_COLLECTION}...")
#             self.client.delete_collection(QDRANT_COLLECTION)
#             time.sleep(5)  # Ensure deletion is processed
#             logger.info(f"✅ Collection {QDRANT_COLLECTION} deleted.")

#         logger.info(f"🚀 Creating new collection: {QDRANT_COLLECTION}...")
#         model = SentenceTransformer(MODEL_NAME)
#         self.client.create_collection(
#             collection_name=QDRANT_COLLECTION,
#             vectors_config=VectorParams(
#                 size=model.get_sentence_embedding_dimension(),
#                 distance=Distance.COSINE
#             )
#         )
#         logger.info(f"✅ Collection {QDRANT_COLLECTION} created.")

#     @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
#     def safe_upsert(self, points: List[PointStruct]):
#         """Retryable upsert function to store embeddings safely."""
#         try:
#             return self.client.upsert(
#                 collection_name=QDRANT_COLLECTION,
#                 points=points,
#                 wait=True
#             )
#         except Exception as e:
#             logger.error(f"Upsert failed: {str(e)}")
#             time.sleep(5)
#             raise

# def sanitize_text(text: str) -> str:
#     """Ensure text is hashable and safe for processing."""
#     return str(text).encode('utf-8', 'replace').decode('utf-8')

# def process_table(wikidb, db_name: str, table_name: str, model) -> PointStruct:
#     """Process a table and generate its embedding."""
#     try:
#         if table_name not in wikidb.list_collection_names():
#             logger.warning(f"Skipping {table_name}: Table not found")
#             return None

#         logger.info(f"Processing {db_name}.{table_name}")
#         table_data = list(wikidb[table_name].find({}, {"_id": 0}))
        
#         # Handle schema data safely
#         schema = list(table_data[0].keys()) if table_data else []

#         # Construct table text representation
#         table_text = " ".join(f"{k}: {v}" for doc in table_data[:3] for k, v in doc.items())
        
#         # Generate embedding safely
#         embedding = model.encode([table_text], batch_size=1, show_progress_bar=False)[0]
        
#         # Create a unique ID
#         table_id = int(hashlib.md5(table_text.encode()).hexdigest()[:16], 16)

#         return PointStruct(
#             id=table_id,
#             vector=embedding.tolist(),
#             payload={
#                 "database": sanitize_text(db_name),
#                 "table": sanitize_text(table_name),
#                 "text": sanitize_text(table_text)  # Store extracted table text
#             }
#         )
#     except Exception as e:
#         logger.error(f"❌ Failed processing {table_name}: {str(e)}")
#         return None

# def store_vanilla_embeddings():
#     """Main function to store embeddings in Qdrant."""
#     qdrant_mgr = QdrantManager()
#     model = SentenceTransformer(MODEL_NAME).to("cpu")

#     # Recreate the collection before inserting embeddings
#     qdrant_mgr.recreate_collection()

#     try:
#         with MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=15000) as mongo_client:
#             wikidb = mongo_client["wikidbs_10k"]
            
#             # Get all database names
#             BASE_PATH = "/datastore/servers/Enas/enas/tabert/wikidbs-10k/databases"
#             all_databases = os.listdir(BASE_PATH)

#             for db_name in all_databases:
#                 tables = [col for col in wikidb.list_collection_names() if col.startswith(db_name)]

#                 # Batch processing for efficiency
#                 batch = []
                
#                 for table_name in tables:
#                     point = process_table(wikidb, db_name, table_name, model)
#                     if point:
#                         batch.append(point)
                        
#                         if len(batch) >= BATCH_SIZE:
#                             qdrant_mgr.safe_upsert(batch)
#                             batch = []
#                             time.sleep(1)  # Rate limiting

#                 # Process any remaining points
#                 if batch:
#                     qdrant_mgr.safe_upsert(batch)

#     except Exception as global_error:
#         logger.error(f"💥 Critical failure: {str(global_error)}")
#         raise

# if __name__ == "__main__":
#     store_vanilla_embeddings()


# import os
# import hashlib
# from pymongo import MongoClient
# from qdrant_client import QdrantClient, models
# from sentence_transformers import SentenceTransformer

# VANILLA_MODEL = "all-mpnet-base-v2"
# QDRANT_VANILLA_COLLECTION = "benchmark_vanilla_sbert"

# def store_vanilla_embeddings():

#     qdrant = QdrantClient("localhost:6333")
#     mongo = MongoClient("mongodb://localhost:27017/")
#     db = mongo["wikidbs_10k"]
#     model = SentenceTransformer(VANILLA_MODEL)
    

#     qdrant.recreate_collection(
#         collection_name=QDRANT_VANILLA_COLLECTION,
#         vectors_config=models.VectorParams(
#             size=model.get_sentence_embedding_dimension(),
#             distance=models.Distance.COSINE
#         )
#     )
    
#     for db_name in os.listdir("/wikidbs-10k/databases"):
#         collections = [col for col in db.list_collection_names() if col.startswith(db_name)]
#         for collection in collections:
#             docs = list(db[collection].find({}, {"_id": 0}))
#             table_text = " ".join(f"{k}: {v}" for doc in docs for k, v in doc.items())
            

#             embedding = model.encode(table_text)
#             qdrant.upsert(
#                 collection_name=QDRANT_VANILLA_COLLECTION,
#                 points=[models.PointStruct(
#                     id=int(hashlib.md5(table_text.encode()).hexdigest()[:16], 16),
#                     vector=embedding.tolist(),
#                     payload={
#                         "db": db_name,
#                         "table": collection,
#                         "text": table_text  
#                     }
#                 )]
#             )

# if __name__ == "__main__":
#     store_vanilla_embeddings()






# for the beauty
# import qdrant_client
# import hashlib
# import time
# from typing import List
# from qdrant_client.models import VectorParams, Distance, PointStruct
# from sentence_transformers import SentenceTransformer
# from pymongo import MongoClient
# from tenacity import retry, stop_after_attempt, wait_exponential
# import logging
# import os

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("vanilla_embedding_store.log"), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# MODEL_NAME = "all-mpnet-base-v2"
# QDRANT_COLLECTION = "benchmark_vanilla_sbert"
# BATCH_SIZE = 50  # Reduced batch size for stability
# TIMEOUT = 60  # Increased timeout to 60 seconds

# class QdrantManager:
#     def __init__(self):
#         self.client = qdrant_client.QdrantClient(
#             "localhost",
#             port=6333,
#             timeout=TIMEOUT,
#             prefer_grpc=False  # Disable gRPC to prevent connection issues
#         )

#     def recreate_collection(self):
#         """Delete the collection if it exists, then recreate it."""
#         if self.client.collection_exists(QDRANT_COLLECTION):
#             logger.info(f"🗑 Deleting existing collection: {QDRANT_COLLECTION}...")
#             self.client.delete_collection(QDRANT_COLLECTION)
#             time.sleep(5)  # Ensure deletion is processed
#             logger.info(f"✅ Collection {QDRANT_COLLECTION} deleted.")

#         logger.info(f"🚀 Creating new collection: {QDRANT_COLLECTION}...")
#         model = SentenceTransformer(MODEL_NAME)
#         self.client.create_collection(
#             collection_name=QDRANT_COLLECTION,
#             vectors_config=VectorParams(
#                 size=model.get_sentence_embedding_dimension(),
#                 distance=Distance.COSINE
#             )
#         )
#         logger.info(f"✅ Collection {QDRANT_COLLECTION} created.")

#     @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
#     def safe_upsert(self, points: List[PointStruct]):
#         """Retryable upsert function to store embeddings safely."""
#         try:
#             return self.client.upsert(
#                 collection_name=QDRANT_COLLECTION,
#                 points=points,
#                 wait=True
#             )
#         except Exception as e:
#             logger.error(f"Upsert failed: {str(e)}")
#             time.sleep(5)
#             raise

# def sanitize_text(text: str) -> str:
#     """Ensure text is hashable and safe for processing."""
#     return str(text).encode('utf-8', 'replace').decode('utf-8')

# def process_table(wikidb, db_name: str, table_name: str, model) -> PointStruct:
#     """Process a table and generate its embedding."""
#     try:
#         if table_name not in wikidb.list_collection_names():
#             logger.warning(f"Skipping {table_name}: Table not found")
#             return None

#         logger.info(f"Processing {db_name}.{table_name}")
#         table_data = list(wikidb[table_name].find({}, {"_id": 0}))
        
#         # Handle schema data safely
#         schema = list(table_data[0].keys()) if table_data else []

#         # Construct table text representation
#         table_text = " ".join(f"{k}: {v}" for doc in table_data[:3] for k, v in doc.items())
        
#         # Generate embedding safely
#         embedding = model.encode([table_text], batch_size=1, show_progress_bar=False)[0]
        
#         # Create a unique ID
#         table_id = int(hashlib.md5(table_text.encode()).hexdigest()[:16], 16)

#         return PointStruct(
#             id=table_id,
#             vector=embedding.tolist(),
#             payload={
#                 "database": sanitize_text(db_name),
#                 "table": sanitize_text(table_name),
#                 "text": sanitize_text(table_text)  # Store extracted table text
#             }
#         )
#     except Exception as e:
#         logger.error(f"❌ Failed processing {table_name}: {str(e)}")
#         return None

# def store_vanilla_embeddings():
#     """Main function to store embeddings in Qdrant."""
#     qdrant_mgr = QdrantManager()
#     model = SentenceTransformer(MODEL_NAME).to("cuda")

#     # Recreate the collection before inserting embeddings
#     qdrant_mgr.recreate_collection()

#     try:
#         with MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=15000) as mongo_client:
#             wikidb = mongo_client["wikidbs_10k"]
            
#             # Get all database names
#             BASE_PATH = "/wikidbs-10k/databases"
#             all_databases = os.listdir(BASE_PATH)

#             for db_name in all_databases:
#                 tables = [col for col in wikidb.list_collection_names() if col.startswith(db_name)]

#                 # Batch processing for efficiency
#                 batch = []
                
#                 for table_name in tables:
#                     point = process_table(wikidb, db_name, table_name, model)
#                     if point:
#                         batch.append(point)
                        
#                         if len(batch) >= BATCH_SIZE:
#                             qdrant_mgr.safe_upsert(batch)
#                             batch = []
#                             time.sleep(1)  # Rate limiting

#                 # Process any remaining points
#                 if batch:
#                     qdrant_mgr.safe_upsert(batch)

#     except Exception as global_error:
#         logger.error(f"💥 Critical failure: {str(global_error)}")
#         raise

# if __name__ == "__main__":
#     store_vanilla_embeddings()


# # import os
# # import hashlib
# # from pymongo import MongoClient
# # from qdrant_client import QdrantClient, models
# # from sentence_transformers import SentenceTransformer

# # VANILLA_MODEL = "all-mpnet-base-v2"
# # QDRANT_VANILLA_COLLECTION = "benchmark_vanilla_sbert"

# # def store_vanilla_embeddings():

# #     qdrant = QdrantClient("localhost:6333")
# #     mongo = MongoClient("mongodb://localhost:27017/")
# #     db = mongo["wikidbs_10k"]
# #     model = SentenceTransformer(VANILLA_MODEL)
    

# #     qdrant.recreate_collection(
# #         collection_name=QDRANT_VANILLA_COLLECTION,
# #         vectors_config=models.VectorParams(
# #             size=model.get_sentence_embedding_dimension(),
# #             distance=models.Distance.COSINE
# #         )
# #     )
    
# #     for db_name in os.listdir("/wikidbs-10k/databases"):
# #         collections = [col for col in db.list_collection_names() if col.startswith(db_name)]
# #         for collection in collections:
# #             docs = list(db[collection].find({}, {"_id": 0}))
# #             table_text = " ".join(f"{k}: {v}" for doc in docs for k, v in doc.items())
            

# #             embedding = model.encode(table_text)
# #             qdrant.upsert(
# #                 collection_name=QDRANT_VANILLA_COLLECTION,
# #                 points=[models.PointStruct(
# #                     id=int(hashlib.md5(table_text.encode()).hexdigest()[:16], 16),
# #                     vector=embedding.tolist(),
# #                     payload={
# #                         "db": db_name,
# #                         "table": collection,
# #                         "text": table_text  
# #                     }
# #                 )]
# #             )

# # if __name__ == "__main__":
# #     store_vanilla_embeddings()
