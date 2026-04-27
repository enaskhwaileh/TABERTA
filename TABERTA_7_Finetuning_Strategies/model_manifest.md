# TABERTA Model Manifest

This repository contains fine-tuning code only. Keep trained model weights out of Git and publish them through Hugging Face.

| Strategy | Local default output | Suggested Hugging Face model ID |
| --- | --- | --- |
| PC | `models/PC_pairwise_contrastive` | `enaskhwaileh/taberta-pc-pairwise-contrastive` |
| SS-C | `models/SSC_simcse` | `enaskhwaileh/taberta-ssc-simcse` |
| TC | `models/TC_triplet_contrastive` | `enaskhwaileh/taberta-tc-triplet-contrastive` |
| TC-Opt | `models/TC_Opt_triplet_optimized` | `enaskhwaileh/taberta-tc-opt-triplet-optimized` |
| TC-SB | `models/TC_SB_triplet_smartbatch` | `enaskhwaileh/taberta-tc-sb-smartbatch` |
| MLM | `models/MLM_masked_lm` | `enaskhwaileh/taberta-mlm-masked-lm` |
| Hybrid | `models/Hybrid_mlm_tc` | `enaskhwaileh/taberta-hybrid-mlm-tc` |

Update the Hugging Face IDs after the final model cards are published.
