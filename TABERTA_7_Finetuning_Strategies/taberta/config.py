"""
Training strategy configurations — one dataclass per paper strategy.

Each config encodes the hyperparameters, serialization view(s), loss,
supervision type, and base model as described in Table 1 of the paper.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class TrainingStrategy(str, Enum):
    PC = "PC"
    SSC = "SS-C"
    TC = "TC"
    TC_OPT = "TC-Opt"
    TC_SB = "TC-SB"
    MLM = "MLM"
    HYBRID = "Hybrid"


@dataclass
class _BaseConfig:
    """Shared defaults across all strategies."""
    base_model: str = "all-mpnet-base-v2"
    batch_size: int = 16
    learning_rate: float = 2e-5
    row_limit: int = 3
    warmup_steps: int = 100
    device: str = "cuda"
    seed: int = 42


@dataclass
class PCConfig(_BaseConfig):
    """Pairwise Contrastive (PC) — InfoNCE, RowView, supervised."""
    strategy: TrainingStrategy = TrainingStrategy.PC
    views: List[str] = field(default_factory=lambda: ["row"])
    epochs: int = 50
    loss: str = "CosineSimilarityLoss"
    supervision: str = "supervised"
    patience: int = 5
    output_dir: str = "models/PC_pairwise_contrastive"


@dataclass
class SSCConfig(_BaseConfig):
    """Self-Supervised Contrastive / SimCSE (SS-C) — in-batch negatives, FullView."""
    strategy: TrainingStrategy = TrainingStrategy.SSC
    views: List[str] = field(default_factory=lambda: ["full"])
    epochs: int = 50
    loss: str = "MultipleNegativesRankingLoss"
    supervision: str = "self-supervised"
    patience: int = 5
    output_dir: str = "models/SSC_simcse"


@dataclass
class TCConfig(_BaseConfig):
    """Triplet Contrastive (TC) — margin-based triplet loss, FullView."""
    strategy: TrainingStrategy = TrainingStrategy.TC
    views: List[str] = field(default_factory=lambda: ["full"])
    epochs: int = 50
    loss: str = "TripletLoss"
    supervision: str = "supervised"
    target_triplets: int = 10000
    patience: int = 5
    output_dir: str = "models/TC_triplet_contrastive"


@dataclass
class TCOptConfig(_BaseConfig):
    """Triplet Contrastive Optimized (TC-Opt) — TripletLoss + AMP, FullView+SchemaView."""
    strategy: TrainingStrategy = TrainingStrategy.TC_OPT
    views: List[str] = field(default_factory=lambda: ["full", "schema"])
    epochs: int = 50
    loss: str = "TripletLoss"
    supervision: str = "supervised"
    target_triplets: int = 10000
    use_amp: bool = True
    gradient_checkpointing: bool = True
    patience: int = 5
    output_dir: str = "models/TC_Opt_triplet_optimized"


@dataclass
class TCSBConfig(_BaseConfig):
    """Triplet Contrastive SmartBatch (TC-SB) — TripletLoss + smart batching, FullView."""
    strategy: TrainingStrategy = TrainingStrategy.TC_SB
    views: List[str] = field(default_factory=lambda: ["full"])
    epochs: int = 50
    loss: str = "TripletLoss"
    supervision: str = "supervised"
    target_triplets: int = 10000
    patience: int = 5
    output_dir: str = "models/TC_SB_triplet_smartbatch"


@dataclass
class MLMConfig(_BaseConfig):
    """Masked Language Modeling (MLM) — MLM loss, FullView masked."""
    strategy: TrainingStrategy = TrainingStrategy.MLM
    # MLM uses the raw MPNet model (needs MLM head), not sentence-transformers
    base_model: str = "microsoft/mpnet-base"
    views: List[str] = field(default_factory=lambda: ["full"])
    epochs: int = 50
    loss: str = "MLM"
    supervision: str = "self-supervised"
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    patience: int = 5
    output_dir: str = "models/MLM_masked_lm"


@dataclass
class HybridConfig(_BaseConfig):
    """Hybrid (MLM + TC) — staged: MLM then TripletLoss, FullView → SchemaView+FullView."""
    strategy: TrainingStrategy = TrainingStrategy.HYBRID
    views: List[str] = field(default_factory=lambda: ["full", "schema"])
    epochs: int = 50
    loss: str = "Hybrid_MLM_then_TripletLoss"
    supervision: str = "hybrid"
    target_triplets: int = 10000
    patience: int = 5
    mlm_model_path: str = "models/MLM_masked_lm"
    output_dir: str = "models/Hybrid_mlm_tc"


# Quick lookup by strategy name
STRATEGY_CONFIGS = {
    TrainingStrategy.PC: PCConfig,
    TrainingStrategy.SSC: SSCConfig,
    TrainingStrategy.TC: TCConfig,
    TrainingStrategy.TC_OPT: TCOptConfig,
    TrainingStrategy.TC_SB: TCSBConfig,
    TrainingStrategy.MLM: MLMConfig,
    TrainingStrategy.HYBRID: HybridConfig,
}
