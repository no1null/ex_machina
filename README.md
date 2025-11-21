# Jira Epic Clustering - Advanced ML Edition

Automatically generate Epic groupings for Jira issues using state-of-the-art machine learning models.

## 🚀 Features

- **Heavy-Duty Models**: Support for models up to 1.5B parameters
- **GPU Acceleration**: Automatic CUDA (NVIDIA) and MPS (Apple Silicon) detection
- **Advanced Clustering**: HDBSCAN, BERTopic, KMeans with silhouette optimization
- **Smart Keyword Extraction**: KeyBERT for contextualized keywords
- **Dimensionality Reduction**: UMAP for better semantic clustering
- **Visualizations**: Automatic UMAP and t-SNE cluster plots

## 📦 Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install "numpy<2" pandas scikit-learn sentence-transformers \
            hdbscan umap-learn bertopic keybert matplotlib seaborn \
            openpyxl xlsxwriter
```

## 🎯 Quick Start

### 1. Check Your Hardware
```bash
python check_hardware.py
```

This will show:
- Your GPU type (NVIDIA CUDA / Apple Silicon MPS / CPU)
- Recommended models for your hardware
- Optimal batch sizes
- Example commands

### 2. Run Clustering

**For Apple Silicon (M1/M2/M3) - Recommended:**
```bash
python jira_epic_clustering_v1.py \
    --input Jira_Export.xlsx \
    --output Jira_Export_with_epics.xlsx \
    --model-name BAAI/bge-large-en-v1.5 \
    --method hdbscan \
    --batch-size 16 \
    --visualize
```

**For NVIDIA GPU:**
```bash
python jira_epic_clustering_v1.py \
    --input Jira_Export.xlsx \
    --output Jira_Export_with_epics.xlsx \
    --model-name BAAI/bge-large-en-v1.5 \
    --method hdbscan \
    --batch-size 32 \
    --visualize
```

**For CPU (Fast & Lightweight):**
```bash
python jira_epic_clustering_v1.py \
    --input Jira_Export.xlsx \
    --output Jira_Export_with_epics.xlsx \
    --model-name all-MiniLM-L6-v2 \
    --method hdbscan \
    --visualize
```

## 🎓 Model Selection

### Recommended Models by Hardware

| Hardware | Model | Size | Speed | Quality |
|----------|-------|------|-------|---------|
| **Apple Silicon** | `BAAI/bge-large-en-v1.5` | 335M | Fast | ⭐⭐⭐⭐⭐ |
| **NVIDIA GPU 16GB+** | `sentence-transformers/gtr-t5-xl` | 1.2B | Medium | ⭐⭐⭐⭐⭐ |
| **NVIDIA GPU 8GB+** | `BAAI/bge-large-en-v1.5` | 335M | Fast | ⭐⭐⭐⭐⭐ |
| **CPU** | `all-MiniLM-L6-v2` | 80M | Very Fast | ⭐⭐⭐ |

See [MODEL_GUIDE.md](MODEL_GUIDE.md) for detailed comparisons.

## 🔧 Advanced Options

### Clustering Methods

**HDBSCAN (Recommended)**
- Automatically determines number of clusters
- Handles outliers intelligently
```bash
--method hdbscan
```

**BERTopic**
- Topic modeling with interpretable labels
- Great for exploration
```bash
--method bertopic
```

**KMeans**
- Traditional clustering
- Specify cluster count or auto-optimize
```bash
--method kmeans --optimize-k
```

### Dimensionality Reduction

**With UMAP (Default - Better Quality)**
```bash
--use-umap
```

**Without UMAP (Faster)**
```bash
--no-umap
```

### Visualization

Generate cluster visualizations (UMAP and t-SNE plots):
```bash
--visualize
```

Plots saved to: `cluster_visualizations/`

## 📊 Output

The script generates an Excel file with two sheets:

1. **Issues_Updated**: Original data + two new columns:
   - `Suggested Epic ID`: E1, E2, E3, etc.
   - `Suggested Epic Name`: Epic: keyword1 / keyword2 / keyword3

2. **Epics**: Summary of each proposed Epic:
   - Epic ID
   - Epic Name
   - Cluster ID
   - Issue Count
   - Theme Keywords
   - Representative Summaries

## 💡 Performance Tips

### For Apple Silicon
- Use `BAAI/bge-large-en-v1.5` (optimized for M-series)
- Batch size: 16-32
- Expected: ~100-200 issues/second

### For NVIDIA GPU
- Large VRAM (16GB+): `gtr-t5-xl` or `instructor-xl`
- Medium VRAM (8GB+): `BAAI/bge-large-en-v1.5`
- Batch size: 32-64

### For CPU
- Use `all-MiniLM-L6-v2`
- Batch size: 8-16
- Consider cloud GPU for large datasets

## 🔍 Example Commands

### Best Quality (Requires GPU)
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name sentence-transformers/gtr-t5-xl \
    --batch-size 12 \
    --method hdbscan \
    --visualize
```

### Multilingual Support
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name paraphrase-multilingual-mpnet-base-v2 \
    --method hdbscan \
    --visualize
```

### Fast Iteration (CPU)
```bash
python jira_epic_clustering_v1.py \
    --input data.xlsx \
    --output output.xlsx \
    --model-name all-MiniLM-L6-v2 \
    --batch-size 8 \
    --method kmeans \
    --k 8
```

## 🚨 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
--batch-size 8

# Or use smaller model
--model-name all-mpnet-base-v2
```

### Slow Performance
```bash
# Check GPU detection
python check_hardware.py

# Use lightweight model
--model-name all-MiniLM-L6-v2
```

### Poor Clustering Results
```bash
# Use larger model
--model-name BAAI/bge-large-en-v1.5

# Try different method
--method bertopic

# Ensure UMAP enabled (default)
--use-umap
```

## 📚 Files

- `jira_epic_clustering_v1.py` - Main clustering script
- `check_hardware.py` - Hardware detection and recommendations
- `MODEL_GUIDE.md` - Comprehensive model comparison guide
- `README.md` - This file

## 🔬 Technical Details

### Supported Models
- Lightweight: 33M - 80M parameters
- Medium: 110M - 278M parameters  
- Large: 335M - 355M parameters
- Heavy-duty: 1.2B - 1.5B parameters

### GPU Support
- NVIDIA CUDA (automatic detection)
- Apple Silicon MPS (automatic detection)
- CPU fallback

### Clustering Algorithms
- HDBSCAN: Density-based, auto clusters
- BERTopic: Topic modeling with interpretability
- KMeans: Traditional with silhouette optimization

### Dimensionality Reduction
- UMAP: Preserves local/global structure
- t-SNE: Visualization only

## 📈 Benchmarks

Approximate processing times (1000 issues):

| Hardware | Model | Time |
|----------|-------|------|
| Apple M2 | BGE-large | ~2 min |
| NVIDIA 3090 | BGE-large | ~1 min |
| NVIDIA 3090 | GTR-T5-XL | ~3 min |
| CPU (i7) | MiniLM | ~5 min |

## 📄 License

MIT

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
