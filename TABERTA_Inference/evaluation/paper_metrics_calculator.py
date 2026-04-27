"""
Metrics Calculation for Paper Tables
Matches evaluation methodology from TABERTA paper
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict

class PaperMetricsCalculator:
    """
    Calculate metrics exactly as reported in paper tables:
    - Table 3: WikiTables (MAP, MRR, NDCG@5/10/20/60)
    - Table 4: TabFact (Precision, Recall, F1)
    - Table 5: FeTaQA/OTTQA (Recall@k, BLEU)
    - Table 6-7: Spider/BIRD (CR@k, MRR@10, NDCG@10)
    """
    
    @staticmethod
    def calculate_map(retrieved_lists: List[List[str]], 
                      relevant_sets: List[Set[str]]) -> float:
        """
        Mean Average Precision (for Table 3 - WikiTables)
        """
        aps = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            if not relevant:
                continue
            
            precisions = []
            num_relevant = 0
            
            for k, item in enumerate(retrieved, 1):
                if item in relevant:
                    num_relevant += 1
                    precision_at_k = num_relevant / k
                    precisions.append(precision_at_k)
            
            if precisions:
                ap = sum(precisions) / len(relevant)
            else:
                ap = 0.0
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def calculate_mrr(retrieved_lists: List[List[str]], 
                      relevant_sets: List[Set[str]]) -> float:
        """
        Mean Reciprocal Rank (for Table 3, 6, 7)
        """
        rrs = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            rr = 0.0
            for rank, item in enumerate(retrieved, 1):
                if item in relevant:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)
        
        return np.mean(rrs)
    
    @staticmethod
    def calculate_ndcg(retrieved_lists: List[List[str]], 
                       relevant_sets: List[Set[str]], 
                       k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain (for Table 3, 6, 7)
        """
        ndcgs = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            # DCG
            dcg = sum([
                1.0 / np.log2(i + 2) 
                if retrieved[i] in relevant else 0.0
                for i in range(min(k, len(retrieved)))
            ])
            
            # IDCG
            idcg = sum([
                1.0 / np.log2(i + 2) 
                for i in range(min(k, len(relevant)))
            ])
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    @staticmethod
    def calculate_precision_recall_f1(retrieved_lists: List[List[str]], 
                                      relevant_sets: List[Set[str]], 
                                      k: int = 5) -> Dict[str, float]:
        """
        Precision, Recall, F1 @ k (for Table 4 - TabFact)
        """
        precisions, recalls, f1s = [], [], []
        
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            top_k = set(retrieved[:k])
            tp = len(top_k & relevant)
            
            precision = tp / k if k > 0 else 0.0
            recall = tp / len(relevant) if relevant else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            "Precision": np.mean(precisions),
            "Recall": np.mean(recalls),
            "F1": np.mean(f1s)
        }
    
    @staticmethod
    def calculate_recall_at_k(retrieved_lists: List[List[str]], 
                              relevant_sets: List[Set[str]], 
                              k: int = 10) -> float:
        """
        Recall@k (for Table 5 - FeTaQA/OTTQA)
        """
        recalls = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            top_k = set(retrieved[:k])
            recall = len(top_k & relevant) / len(relevant) if relevant else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    @staticmethod
    def calculate_containment_recall(retrieved_lists: List[List[str]], 
                                    gold_schemas: List[List[str]], 
                                    k: int = 10) -> float:
        """
        Containment Recall@k (for Table 6-7 - Spider/BIRD)
        Checks if all gold tables are in top-k retrieved
        """
        crs = []
        for retrieved, gold in zip(retrieved_lists, gold_schemas):
            top_k = set(retrieved[:k])
            gold_set = set(gold)
            
            # CR = 1 if all gold tables are retrieved, 0 otherwise
            cr = 1.0 if gold_set.issubset(top_k) else 0.0
            crs.append(cr)
        
        return np.mean(crs)


def generate_paper_table_3(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate Table 3 format: WikiTables Ad-hoc Retrieval
    
    Input: model_results = {
        "model_name": {
            "retrieved": [[list of retrieved tables per query]],
            "relevant": [[set of relevant tables per query]]
        }
    }
    """
    calc = PaperMetricsCalculator()
    
    results = []
    for model_name, data in model_results.items():
        retrieved = data["retrieved"]
        relevant = data["relevant"]
        
        row = {
            "Model": model_name,
            "MAP": calc.calculate_map(retrieved, relevant),
            "MRR": calc.calculate_mrr(retrieved, relevant),
            "NDCG@5": calc.calculate_ndcg(retrieved, relevant, k=5),
            "NDCG@10": calc.calculate_ndcg(retrieved, relevant, k=10),
            "NDCG@20": calc.calculate_ndcg(retrieved, relevant, k=20),
            "NDCG@60": calc.calculate_ndcg(retrieved, relevant, k=60),
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def generate_paper_table_4(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate Table 4 format: TabFact Evidence Retrieval
    """
    calc = PaperMetricsCalculator()
    
    results = []
    for model_name, data in model_results.items():
        retrieved = data["retrieved"]
        relevant = data["relevant"]
        
        metrics = calc.calculate_precision_recall_f1(retrieved, relevant, k=5)
        
        row = {
            "Model": model_name,
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1"]
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def generate_paper_table_5(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate Table 5 format: FeTaQA/OTTQA Retrieval
    """
    calc = PaperMetricsCalculator()
    
    results = []
    for model_name, data in model_results.items():
        retrieved = data["retrieved"]
        relevant = data["relevant"]
        
        row = {
            "Model": model_name,
            "R@1": calc.calculate_recall_at_k(retrieved, relevant, k=1),
            "R@5": calc.calculate_recall_at_k(retrieved, relevant, k=5),
            "R@10": calc.calculate_recall_at_k(retrieved, relevant, k=10),
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def generate_paper_table_6_7(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Generate Table 6-7 format: Spider/BIRD Schema Grounding
    """
    calc = PaperMetricsCalculator()
    
    results = []
    for model_name, data in model_results.items():
        retrieved = data["retrieved"]
        gold_schemas = data["gold_schemas"]  # List of lists of gold tables
        
        row = {
            "Model": model_name,
            "CR@1": calc.calculate_containment_recall(retrieved, gold_schemas, k=1),
            "CR@5": calc.calculate_containment_recall(retrieved, gold_schemas, k=5),
            "CR@10": calc.calculate_containment_recall(retrieved, gold_schemas, k=10),
            "MRR@10": calc.calculate_mrr(retrieved, [set(g) for g in gold_schemas]),
            "NDCG@10": calc.calculate_ndcg(retrieved, [set(g) for g in gold_schemas], k=10),
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def generate_decoder_comparison_table(
    decoder_results: Dict,
    taberta_results: Dict
) -> pd.DataFrame:
    """
    Generate new table for R3D4: Decoder vs Bidirectional Comparison
    
    Format for paper revision
    """
    
    all_models = {}
    all_models.update(decoder_results)
    all_models.update(taberta_results)
    
    calc = PaperMetricsCalculator()
    
    results = []
    for model_name, data in all_models.items():
        # Determine model type
        if any(x in model_name.lower() for x in ["qwen", "mistral", "decoder", "e5-", "bge"]):
            if "bge" in model_name.lower():
                model_type = "Hybrid"
                params = "335M"
            else:
                model_type = "Decoder-Only"
                params = "7B"
        else:
            model_type = "Bidirectional"
            params = "110M"
        
        retrieved = data["retrieved"]
        relevant = data["relevant"]
        
        row = {
            "Model": model_name,
            "Architecture": model_type,
            "Parameters": params,
            "Recall@5": calc.calculate_recall_at_k(retrieved, relevant, k=5),
            "Recall@10": calc.calculate_recall_at_k(retrieved, relevant, k=10),
            "MRR": calc.calculate_mrr(retrieved, relevant),
            "NDCG@10": calc.calculate_ndcg(retrieved, relevant, k=10),
            "Inference (ms)": data.get("inference_time_ms", "N/A")
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Sort by architecture (Decoder-only first, then Hybrid, then Bidirectional)
    df["sort_key"] = df["Architecture"].map({
        "Decoder-Only": 1, "Hybrid": 2, "Bidirectional": 3
    })
    df = df.sort_values("sort_key").drop("sort_key", axis=1)
    
    return df


# ======== Usage Example ========
if __name__ == "__main__":
    # Example: Format results for paper
    
    # Simulated results structure
    example_results = {
        "Decoder-Qwen2-7B": {
            "retrieved": [["table1", "table2", "table3", ...], ...],  # Per query
            "relevant": [{"table1", "table3"}, ...],  # Per query
            "inference_time_ms": 120
        },
        "TABERTA-Hybrid": {
            "retrieved": [["table1", "table4", "table2", ...], ...],
            "relevant": [{"table1", "table3"}, ...],
            "inference_time_ms": 8
        },
        # ... more models
    }
    
    # Generate comparison table
    decoder_results = {"Decoder-Qwen2-7B": example_results["Decoder-Qwen2-7B"]}
    taberta_results = {"TABERTA-Hybrid": example_results["TABERTA-Hybrid"]}
    
    comparison_df = generate_decoder_comparison_table(decoder_results, taberta_results)
    
    print("\nDecoder vs Bidirectional Comparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Save for LaTeX
    comparison_df.to_latex("decoder_comparison.tex", index=False, float_format="%.4f")
    comparison_df.to_csv("decoder_comparison.csv", index=False)
    
    print("\n✓ Tables saved to decoder_comparison.tex and decoder_comparison.csv")
    print("  Import into your paper's Table X (R3D4 response)")
