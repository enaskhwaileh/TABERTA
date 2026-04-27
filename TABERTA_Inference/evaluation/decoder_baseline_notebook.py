"""
Quick Decoder-Only Baseline Evaluation
Add this to your evaluation.ipynb as a new cell
"""

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
from rich.console import Console
from rich.table import Table as RichTable
from rich import box
from tqdm import tqdm

# ======== CONFIGURATION ========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
console = Console()

# Decoder-only model choices (pick one based on availability)
DECODER_MODELS = {
    # Option 1: If you have access to newer models
    "qwen2-7b": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    
    # Option 2: E5-Mistral (good alternative)
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",
    
    # Option 3: BGE (smaller, faster)
    "bge-large": "BAAI/bge-large-en-v1.5",
}

# Select which model to use
SELECTED_MODEL = "bge-large"  # Change this to test different models

# ======== STEP 1: Load Decoder Model ========
console.print(f"[yellow]Loading {SELECTED_MODEL}...[/yellow]")
decoder_model = SentenceTransformer(DECODER_MODELS[SELECTED_MODEL], device=DEVICE)
console.print("[green]✓ Model loaded[/green]")

# ======== STEP 2: Serialize Tables (Same as Your Paper) ========
def serialize_full_view_simple(table_name: str, sample_docs: List[Dict]) -> str:
    """Simplified FullView serialization"""
    if not sample_docs:
        return f"[TABLE] {table_name}"
    
    # Get column names
    cols = [k for k in sample_docs[0].keys() if k != '_id']
    schema = ", ".join(cols)
    
    # Get sample rows
    rows = []
    for doc in sample_docs[:5]:  # Max 5 rows
        row_vals = [str(doc.get(col, "")) for col in cols]
        rows.append(" | ".join(row_vals))
    
    rows_str = " ".join(rows)
    return f"[TABLE] {table_name} [SCHEMA] {schema} [ROWS] {rows_str}"

# ======== STEP 3: Generate & Store Embeddings ========
def generate_and_store_decoder_embeddings(
    model,
    collection_name: str,
    max_tables: int = 100,  # Set to None for all tables
    batch_size: int = 8
):
    """Generate embeddings for tables and store in Qdrant"""
    
    # Get tables
    table_names = wikidb.list_collection_names()
    if max_tables:
        table_names = table_names[:max_tables]
    
    console.print(f"[cyan]Processing {len(table_names)} tables...[/cyan]")
    
    # Generate embeddings in batches
    all_texts = []
    all_table_names = []
    
    for table_name in tqdm(table_names, desc="Serializing"):
        try:
            docs = list(wikidb[table_name].find().limit(5))
            if docs:
                text = serialize_full_view_simple(table_name, docs)
                all_texts.append(text)
                all_table_names.append(table_name)
        except Exception as e:
            continue
    
    # Encode all at once
    console.print("[yellow]Encoding tables...[/yellow]")
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Store in Qdrant
    console.print("[yellow]Storing in Qdrant...[/yellow]")
    
    # Create collection
    try:
        qdrant_client.delete_collection(collection_name)
    except:
        pass
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    
    # Create points
    points = [
        PointStruct(
            id=idx,
            vector=emb.tolist(),
            payload={"table_name": name, "table": name}
        )
        for idx, (name, emb) in enumerate(zip(all_table_names, embeddings))
    ]
    
    # Upload
    qdrant_client.upsert(collection_name=collection_name, points=points)
    console.print(f"[green]✓ Stored {len(points)} embeddings[/green]")

# ======== STEP 4: Evaluate Against Your Baselines ========
def quick_comparison_test(query_table_name: str = None):
    """
    Quick test: Compare decoder vs your best models on a single query
    """
    
    # If no query table provided, pick random
    if not query_table_name:
        query_table_name = np.random.choice(wikidb.list_collection_names())
    
    # Get query table
    query_docs = list(wikidb[query_table_name].find().limit(5))
    query_text = serialize_full_view_simple(query_table_name, query_docs)
    
    console.print(f"\n[bold magenta]Query Table:[/bold magenta] {query_table_name}")
    
    # Models to compare
    models_to_test = {
        f"Decoder ({SELECTED_MODEL})": {
            "model": decoder_model,
            "collection": f"decoder_{SELECTED_MODEL}"
        },
        "TABERTA Hybrid": {
            "model": SentenceTransformer("7_hybrid_model_reg").to(DEVICE),
            "collection": "full_hybrid"
        },
        "TABERTA Supervised": {
            "model": SentenceTransformer("5_full_supervised_model").to(DEVICE),
            "collection": "full_supervised"
        },
        "Vanilla SBERT": {
            "model": SentenceTransformer("all-mpnet-base-v2").to(DEVICE),
            "collection": "benchmark_vanilla_sbert"
        }
    }
    
    # Results table
    results_table = RichTable(
        title="[bold]Decoder-Only vs Bidirectional Comparison[/bold]",
        box=box.DOUBLE_EDGE
    )
    results_table.add_column("Model", style="cyan")
    results_table.add_column("Type", style="yellow")
    results_table.add_column("Top-1", style="green")
    results_table.add_column("Avg Score", style="magenta")
    results_table.add_column("Params", style="dim")
    
    for model_name, config in models_to_test.items():
        # Encode query
        query_emb = config["model"].encode([query_text])[0]
        
        # Search
        try:
            results = qdrant_client.search(
                collection_name=config["collection"],
                query_vector=query_emb.tolist(),
                limit=10
            )
            
            top1 = results[0].payload.get("table_name", results[0].payload.get("table"))
            avg_score = np.mean([r.score for r in results[:5]])
            
            # Determine type and params
            if "Decoder" in model_name:
                model_type = "Decoder-Only"
                params = "335M-7B"
            else:
                model_type = "Bidirectional"
                params = "110M-335M"
            
            results_table.add_row(
                model_name,
                model_type,
                top1[:30] + "..." if len(top1) > 30 else top1,
                f"{avg_score:.4f}",
                params
            )
        except Exception as e:
            console.print(f"[red]Error with {model_name}: {e}[/red]")
    
    console.print(results_table)

# ======== STEP 5: Full Benchmark Evaluation ========
def full_benchmark_evaluation(num_queries: int = 50):
    """
    Run full evaluation comparing all models
    Returns metrics DataFrame
    """
    
    import pandas as pd
    
    # Sample random query tables
    all_tables = wikidb.list_collection_names()
    test_tables = np.random.choice(all_tables, size=min(num_queries, len(all_tables)), replace=False)
    
    models = {
        f"Decoder-{SELECTED_MODEL}": {
            "model": decoder_model,
            "collection": f"decoder_{SELECTED_MODEL}"
        },
        "TABERTA-Hybrid": {
            "model": SentenceTransformer("7_hybrid_model_reg").to(DEVICE),
            "collection": "full_hybrid"
        },
        "TABERTA-Supervised": {
            "model": SentenceTransformer("5_full_supervised_model").to(DEVICE),
            "collection": "full_supervised"
        },
        "Vanilla-SBERT": {
            "model": SentenceTransformer("all-mpnet-base-v2").to(DEVICE),
            "collection": "benchmark_vanilla_sbert"
        }
    }
    
    metrics = {name: {"MRR": [], "Recall@5": [], "Recall@10": []} for name in models}
    
    for query_table in tqdm(test_tables, desc="Evaluating"):
        # Get query
        query_docs = list(wikidb[query_table].find().limit(5))
        if not query_docs:
            continue
            
        query_text = serialize_full_view_simple(query_table, query_docs)
        
        # Test each model
        for model_name, config in models.items():
            try:
                # Encode
                query_emb = config["model"].encode([query_text])[0]
                
                # Search
                results = qdrant_client.search(
                    collection_name=config["collection"],
                    query_vector=query_emb.tolist(),
                    limit=10
                )
                
                retrieved = [r.payload.get("table_name", r.payload.get("table")) for r in results]
                
                # MRR
                if query_table in retrieved:
                    rank = retrieved.index(query_table) + 1
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0
                
                # Recall
                recall_5 = 1.0 if query_table in retrieved[:5] else 0.0
                recall_10 = 1.0 if query_table in retrieved[:10] else 0.0
                
                metrics[model_name]["MRR"].append(mrr)
                metrics[model_name]["Recall@5"].append(recall_5)
                metrics[model_name]["Recall@10"].append(recall_10)
                
            except Exception as e:
                console.print(f"[red]Error with {model_name}: {e}[/red]")
    
    # Compute averages
    results_df = pd.DataFrame({
        model: {
            "MRR": np.mean(vals["MRR"]),
            "Recall@5": np.mean(vals["Recall@5"]),
            "Recall@10": np.mean(vals["Recall@10"])
        }
        for model, vals in metrics.items()
    }).T
    
    console.print("\n[bold green]Evaluation Results:[/bold green]")
    console.print(results_df)
    
    return results_df

# ======== USAGE INSTRUCTIONS ========
console.print("""
[bold cyan]Decoder-Only Baseline Evaluation[/bold cyan]

[yellow]Step 1:[/yellow] Generate embeddings (one-time, takes ~5-30 min depending on model)
    generate_and_store_decoder_embeddings(
        decoder_model,
        collection_name=f"decoder_{SELECTED_MODEL}",
        max_tables=None  # None = all tables, or set limit for testing
    )

[yellow]Step 2:[/yellow] Quick single-query test
    quick_comparison_test()

[yellow]Step 3:[/yellow] Full benchmark evaluation
    results = full_benchmark_evaluation(num_queries=100)
    results.to_csv(f"decoder_comparison_{SELECTED_MODEL}.csv")

[green]Note:[/green] Results will show if decoder-only models are competitive
with your fine-tuned bidirectional encoders.
""")
