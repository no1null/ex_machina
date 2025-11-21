# Heavy-Duty Model Guide for Jira Epic Clustering

## 🚀 Quick Start

Check your hardware capabilities:
```bash
source venv/bin/activate
python check_hardware.py
```

## 📊 Model Comparison

### Tier 1: Heavy-Duty Models (1B+ parameters)
**Best for: Maximum quality, when you have powerful GPU (16GB+ VRAM)**

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `hkunlp/instructor-xl` | 1.5B | Task-specific instructions | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| `sentence-transformers/gtr-t5-xl` | 1.2B | Overall best quality | ⭐⭐ | ⭐⭐⭐⭐⭐ |

**Usage:**
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name sentence-transformers/gtr-t5-xl \
    --batch-size 12 \
    --method hdbscan \
    --visualize
```

---

### Tier 2: Large Models (300M+ parameters) ⭐ RECOMMENDED
**Best for: Excellent quality with good speed, GPU with 8GB+ VRAM**

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `BAAI/bge-large-en-v1.5` | 335M | **SOTA performance, balanced** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| `intfloat/e5-large-v2` | 335M | Strong all-around | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `sentence-transformers/all-roberta-large-v1` | 355M | Robust, well-tested | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Usage (Recommended for most users):**
```bash
# For NVIDIA GPU
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name BAAI/bge-large-en-v1.5 \
    --batch-size 32 \
    --method hdbscan \
    --visualize

# For Apple Silicon
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name BAAI/bge-large-en-v1.5 \
    --batch-size 16 \
    --method hdbscan \
    --visualize
```

**Why BGE-large is recommended:**
- State-of-the-art on MTEB benchmark
- Fast inference speed
- Excellent semantic understanding
- Well-optimized for clustering tasks
- Works great on Apple Silicon

---

### Tier 3: Medium Models (100M parameters)
**Best for: Good quality, fast speed, any GPU or powerful CPU**

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `all-mpnet-base-v2` | 110M | General purpose, reliable | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `BAAI/bge-base-en-v1.5` | 110M | Speed + quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `paraphrase-multilingual-mpnet-base-v2` | 278M | Multilingual (50+ languages) | ⭐⭐⭐ | ⭐⭐⭐⭐ |

**Usage:**
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name all-mpnet-base-v2 \
    --batch-size 64 \
    --method hdbscan \
    --visualize
```

---

### Tier 4: Lightweight Models (<100M parameters)
**Best for: CPU-only, fast iteration, or when resources are limited**

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| `all-MiniLM-L6-v2` | 80M | **Default, CPU-friendly** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `all-MiniLM-L12-v2` | 33M | Ultra-fast | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Usage:**
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name all-MiniLM-L6-v2 \
    --batch-size 8 \
    --method hdbscan \
    --visualize
```

---

## 🎯 Method Comparison

### HDBSCAN (Recommended) ⭐
- **Automatically** determines number of clusters
- Handles outliers/noise intelligently
- Best for variable-sized clusters
- No need to specify k

```bash
--method hdbscan
```

### BERTopic (Topic Modeling)
- Generates interpretable topic labels
- Great for exploration
- Slightly slower but very insightful

```bash
--method bertopic
```

### KMeans (Traditional)
- Fast and predictable
- Requires specifying number of clusters
- Good when you know how many epics you want

```bash
--method kmeans --k 10
# or let it auto-optimize:
--method kmeans --optimize-k
```

---

## 💡 Performance Tips

### For Apple Silicon (M1/M2/M3)
1. Use `BAAI/bge-large-en-v1.5` - optimized for Apple GPUs
2. Set batch size: 16-32
3. MPS automatically detected and used
4. Expected speed: ~100-200 issues/second

### For NVIDIA GPU
1. Large VRAM (16GB+): Use `gtr-t5-xl` or `instructor-xl`
2. Medium VRAM (8GB+): Use `BAAI/bge-large-en-v1.5`
3. Small VRAM (4GB): Use `all-mpnet-base-v2`
4. Batch size: Start with 32, increase until OOM

### For CPU Only
1. Stick with `all-MiniLM-L6-v2` (default)
2. Lower batch size: 8-16
3. Consider cloud GPU instance for large datasets

### Memory Optimization
If you get "Out of Memory" errors:
```bash
# Reduce batch size
--batch-size 8

# Or use smaller model
--model-name all-mpnet-base-v2
```

---

## 🔬 Advanced Features

### With UMAP Dimensionality Reduction (Default)
```bash
--use-umap  # Already default, improves clustering quality
```

### Without UMAP (Faster but lower quality)
```bash
--no-umap
```

### With Visualizations
```bash
--visualize  # Creates UMAP and t-SNE plots
```

### Optimize Cluster Count (KMeans only)
```bash
--method kmeans --optimize-k  # Uses silhouette analysis
```

---

## 📈 Expected Performance

| Dataset Size | Model | Hardware | Time | Quality |
|--------------|-------|----------|------|---------|
| 100 issues | Mini-L6 | CPU | ~30s | Good |
| 100 issues | BGE-large | Apple M2 | ~15s | Excellent |
| 1000 issues | Mini-L6 | CPU | ~5m | Good |
| 1000 issues | BGE-large | Apple M2 | ~2m | Excellent |
| 1000 issues | BGE-large | NVIDIA 3090 | ~1m | Excellent |
| 10,000 issues | BGE-large | NVIDIA 3090 | ~8m | Excellent |

---

## 🎓 Model Selection Decision Tree

```
Do you have a GPU?
├─ YES
│  ├─ NVIDIA GPU?
│  │  ├─ 16GB+ VRAM? → gtr-t5-xl or instructor-xl (BEST QUALITY)
│  │  ├─ 8-16GB VRAM? → BAAI/bge-large-en-v1.5 ⭐ (RECOMMENDED)
│  │  └─ <8GB VRAM? → all-mpnet-base-v2
│  └─ Apple Silicon?
│     └─ → BAAI/bge-large-en-v1.5 ⭐ (OPTIMIZED FOR M-SERIES)
└─ NO (CPU only)
   └─ → all-MiniLM-L6-v2 (FAST & EFFICIENT)
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
--batch-size 8

# Or use smaller model
--model-name all-mpnet-base-v2
```

### Slow on CPU
```bash
# Use lightweight model
--model-name all-MiniLM-L6-v2 --batch-size 8
```

### Poor clustering results
```bash
# Use larger model
--model-name BAAI/bge-large-en-v1.5

# Ensure UMAP is enabled (default)
--use-umap

# Try different method
--method bertopic
```

---

## 📚 Further Reading

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Model rankings
- [Sentence Transformers Docs](https://www.sbert.net/)
- [BGE Models Paper](https://arxiv.org/abs/2309.07597)
- [HDBSCAN Guide](https://hdbscan.readthedocs.io/)
