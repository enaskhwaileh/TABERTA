import pymongo
import pandas as pd
import json
import random
from tqdm import tqdm
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
from datetime import datetime

# Configuration
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "wikidbs_10k"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_NAME = "all-mpnet-base-v2"
TOP_K = 10
NUM_TABLES = 100
OUTPUT_CSV = "table_similarity_benchmark.csv"
MIN_TABLE_SIZE = 1  # Minimum documents a table must have to be included

def get_valid_tables():
    """Get list of valid tables with at least MIN_TABLE_SIZE documents"""
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    valid_tables = []
    for collection in db.list_collection_names():
        try:
            # Skip tables with None/empty names
            if not collection or not isinstance(collection, str):
                continue
                
            # Skip empty tables
            if db[collection].count_documents({}) < MIN_TABLE_SIZE:
                continue
                
            valid_tables.append(collection)
        except Exception as e:
            print(f"Skipping invalid collection {collection}: {str(e)}")
    
    client.close()
    return valid_tables

def process_tables():
    """Main processing function with robust error handling"""
    print("Initializing connections...")
    valid_tables = get_valid_tables()
    print(f"Found {len(valid_tables)} valid tables")
    
    if len(valid_tables) < NUM_TABLES:
        print(f"Warning: Only {len(valid_tables)} tables meet criteria (needed {NUM_TABLES})")
    
    selected_tables = random.sample(valid_tables, min(NUM_TABLES, len(valid_tables)))
    print(f"Processing {len(selected_tables)} tables")
    
    # Initialize components
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    qdrant = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    model = SentenceTransformer(MODEL_NAME)
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'query_table',
            'query_sample',
            'rank',
            'candidate_table',
            'similarity_score',
            'evidence',
            'candidate_sample',
            'processing_time'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        processed_count = 0
        for table_name in tqdm(selected_tables, desc="Processing tables"):
            start_time = datetime.now()
            
            try:
                # Get sample data (with robust handling)
                sample = list(db[table_name].aggregate([{'$sample': {'size': 3}}]))
                if not sample:
                    print(f"Skipping empty table: {table_name}")
                    continue
                    
                query_df = pd.DataFrame(sample).drop('_id', axis=1, errors='ignore')
                if query_df.empty:
                    print(f"Skipping malformed table: {table_name}")
                    continue
                
                # Generate embedding
                try:
                    embedding = generate_embedding(query_df)
                except Exception as e:
                    print(f"Embedding failed for {table_name}: {str(e)}")
                    continue
                
                # Find similar tables (using modern Qdrant API)
                try:
                    results = qdrant.search(
                        collection_name="wikidbs_vectors",
                        query_vector=embedding.tolist(),
                        limit=TOP_K + 1,
                        with_payload=True
                    )
                except Exception as e:
                    print(f"Qdrant search failed for {table_name}: {str(e)}")
                    continue
                
                # Process results
                similar_tables = []
                for hit in results:
                    if not hit.payload or 'table' not in hit.payload:
                        continue
                        
                    candidate_name = hit.payload['table']
                    if candidate_name == table_name:  # Skip self
                        continue
                        
                    # Verify candidate exists
                    if candidate_name not in valid_tables:
                        continue
                        
                    # Get candidate sample
                    candidate_sample = list(db[candidate_name].aggregate([{'$sample': {'size': 1}}]))
                    if not candidate_sample:
                        continue
                        
                    candidate_df = pd.DataFrame(candidate_sample).drop('_id', axis=1, errors='ignore')
                    if candidate_df.empty:
                        continue
                        
                    evidence = identify_similarity_evidence(db, table_name, candidate_name)
                    
                    similar_tables.append({
                        'candidate_table': candidate_name,
                        'similarity_score': hit.score,
                        'evidence': evidence,
                        'candidate_sample': candidate_df.head(1).to_dict('records')
                    })
                    
                    if len(similar_tables) >= TOP_K:
                        break
                
                # Write results if we found any matches
                if similar_tables:
                    for rank, similar in enumerate(similar_tables, 1):
                        writer.writerow({
                            'query_table': table_name,
                            'query_sample': json.dumps(query_df.head(1).to_dict('records')),
                            'rank': rank,
                            'candidate_table': similar['candidate_table'],
                            'similarity_score': similar['similarity_score'],
                            'evidence': similar['evidence'],
                            'candidate_sample': json.dumps(similar['candidate_sample']),
                            'processing_time': str(datetime.now() - start_time)
                        })
                    processed_count += 1
                else:
                    print(f"No valid similar tables found for {table_name}")
                    
            except Exception as e:
                print(f"Critical error processing {table_name}: {str(e)}")
    
    client.close()
    print(f"Successfully processed {processed_count}/{len(selected_tables)} tables")
    print(f"Results saved to {OUTPUT_CSV}")

def identify_similarity_evidence(db, table1, table2):
    """Robust evidence identification"""
    evidence = []
    try:
        sample1 = db[table1].find_one()
        sample2 = db[table2].find_one()
        
        if sample1 and sample2:
            cols1 = {k for k in sample1.keys() if k != '_id'}
            cols2 = {k for k in sample2.keys() if k != '_id'}
            
            shared_cols = cols1 & cols2
            if shared_cols:
                evidence.append(f"shared_columns:{','.join(shared_cols)}")
                
                # Check for potential foreign keys
                fk_candidates = [col for col in shared_cols 
                               if any(x in col.lower() for x in ['id', 'key', 'code', 'ref'])]
                if fk_candidates:
                    evidence.append(f"potential_fk:{','.join(fk_candidates)}")
                
                # Check for matching values
                for col in shared_cols:
                    val1 = sample1.get(col)
                    val2 = sample2.get(col)
                    if val1 and val2 and str(val1) == str(val2):
                        evidence.append(f"value_match:{col}={val1}")
                        break
    except:
        pass
    
    return " | ".join(evidence) if evidence else "embedding_similarity_only"

def generate_embedding(table_df):
    """Robust embedding generation"""
    cells = table_df.values.flatten().astype(str)
    cell_embeddings = model.encode(cells)
    return np.mean(cell_embeddings, axis=0)

if __name__ == "__main__":
    process_tables()