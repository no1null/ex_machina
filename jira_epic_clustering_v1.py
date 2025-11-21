"""
jira_epic_clustering.py

Use advanced ML (SentenceTransformers + HDBSCAN/BERTopic + UMAP) to automatically 
propose Epics for Jira issues based on their Summary + Description.

Usage:
    python jira_epic_clustering.py \
        --input Jira_Export.xlsx \
        --output Jira_Export_with_epics.xlsx \
        --method hdbscan  # or 'kmeans', 'bertopic'

Requirements (install via pip as needed):
    - pandas
    - numpy
    - scikit-learn
    - sentence-transformers  (preferred, for semantic embeddings)
    - hdbscan               (density-based clustering)
    - umap-learn            (dimensionality reduction)
    - bertopic              (topic modeling)
    - keybert               (advanced keyword extraction)
    - matplotlib            (visualizations)
    - seaborn               (visualizations)
    
Enhancements:
    - HDBSCAN: Automatic cluster detection, handles outliers
    - UMAP: Better dimensionality reduction for semantic data
    - BERTopic: Topic modeling with interpretable results
    - KeyBERT: Contextualized keyword extraction
    - Silhouette analysis: Automatic optimal cluster count
    - Visualizations: UMAP/t-SNE plots of clusters
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

# Try to import SentenceTransformer (advanced semantic embeddings)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    HAS_SENTENCE_TRANSFORMERS = True
    HAS_CUDA = torch.cuda.is_available()
    HAS_MPS = torch.backends.mps.is_available()  # Apple Silicon GPU
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    HAS_CUDA = False
    HAS_MPS = False

# Try to import HDBSCAN (density-based clustering)
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# Try to import UMAP (dimensionality reduction)
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Try to import BERTopic (topic modeling)
try:
    from bertopic import BERTopic
    HAS_BERTOPIC = True
except Exception:
    HAS_BERTOPIC = False

# Try to import KeyBERT (keyword extraction)
try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except Exception:
    HAS_KEYBERT = False

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    HAS_VIZ = True
except Exception:
    HAS_VIZ = False


def find_column(df: pd.DataFrame, candidates):
    """
    Find a column in df whose name (lowercased) contains any of the candidate substrings.
    Returns the column name or None.
    """
    for col in df.columns:
        low = col.lower()
        for cand in candidates:
            if cand in low:
                return col
    return None


def build_text_series(df, summary_col, description_col):
    """
    Concatenate Summary + Description into a single text field per row.
    """
    return (
        df[summary_col].fillna("").astype(str).str.strip()
        + " "
        + df[description_col].fillna("").astype(str).str.strip()
    ).str.strip()


def compute_embeddings(texts, model_name: str = "all-MiniLM-L6-v2", batch_size: int = None):
    """
    Compute sentence embeddings for the given texts using SentenceTransformers.
    Falls back to TF-IDF embeddings if sentence-transformers is not available.
    
    Supports heavy-duty models with GPU acceleration:
    - Automatically detects CUDA (NVIDIA) or MPS (Apple Silicon)
    - Optimizes batch size based on available hardware
    - Supports large models up to 1B+ parameters
    """
    if HAS_SENTENCE_TRANSFORMERS:
        # Determine device
        if HAS_CUDA:
            device = "cuda"
            default_batch = 32
            print(f"[INFO] GPU detected: CUDA (NVIDIA)")
        elif HAS_MPS:
            device = "mps"
            default_batch = 16
            print(f"[INFO] GPU detected: MPS (Apple Silicon)")
        else:
            device = "cpu"
            default_batch = 8
            print(f"[INFO] No GPU detected, using CPU")
        
        if batch_size is None:
            batch_size = default_batch
        
        print(f"[INFO] Loading SentenceTransformer model: {model_name}")
        print(f"[INFO] Device: {device}, Batch size: {batch_size}")
        
        model = SentenceTransformer(model_name, device=device)
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO] Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        
        embeddings = model.encode(
            texts.tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Better for clustering
        )
        return np.array(embeddings), "sentence-transformers"
    else:
        print("[WARN] sentence-transformers not available. Falling back to TF-IDF.")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2000,
        )
        X = vectorizer.fit_transform(texts)
        return X, "tfidf"


def choose_cluster_count(n_items: int, user_k: int = None) -> int:
    """
    Choose a reasonable number of clusters.
    - If user_k is provided, use that.
    - Otherwise, base on sqrt(n_items) and clamp between 2 and 15.
    """
    if user_k is not None and user_k > 1:
        return int(user_k)

    base = max(2, int(round(math.sqrt(max(n_items, 2)) / 1.2)))
    return max(2, min(15, base))


def find_optimal_clusters_silhouette(X, min_k=2, max_k=15):
    """
    Use silhouette analysis to find the optimal number of clusters.
    Returns the k with the best silhouette score.
    """
    print(f"[INFO] Running silhouette analysis to find optimal k...")
    best_k = min_k
    best_score = -1
    
    for k in range(min_k, min(max_k + 1, len(X))):
        if k >= len(X):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, labels)
        print(f"[INFO]   k={k}: silhouette={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"[INFO] Optimal k={best_k} with silhouette score={best_score:.4f}")
    return best_k


def reduce_dimensions_umap(X, n_components=5, n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality using UMAP for better clustering.
    UMAP preserves both local and global structure better than PCA.
    """
    if not HAS_UMAP:
        print("[WARN] UMAP not available, skipping dimensionality reduction.")
        return X
    
    print(f"[INFO] Reducing dimensions with UMAP to {n_components} components...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    X_reduced = reducer.fit_transform(X)
    print(f"[INFO] Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
    return X_reduced


def extract_keywords(texts, top_n=5, use_keybert=True):
    """
    Extract keywords from a list of text snippets.
    If KeyBERT is available and use_keybert=True, uses contextualized extraction.
    Otherwise falls back to frequency-based extraction.
    """
    if use_keybert and HAS_KEYBERT:
        print(f"[INFO] Extracting keywords with KeyBERT (contextualized)")
        kw_model = KeyBERT()
        all_text = " ".join(texts)
        try:
            keywords_scores = kw_model.extract_keywords(
                all_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                diversity=0.5  # MMR for diverse keywords
            )
            keywords = [kw for kw, score in keywords_scores]
            return keywords
        except Exception as e:
            print(f"[WARN] KeyBERT extraction failed: {e}. Falling back to frequency.")
    
    # Fallback: frequency-based extraction
    stopwords = set(ENGLISH_STOP_WORDS)
    domain_stop = {
        "lrt", "jira", "request", "requests", "story", "task",
        "bug", "issue", "sprint", "epic", "project"
    }

    all_text = " ".join(texts).lower()
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]+\b", all_text)
    freq = Counter(tokens)
    keywords = []
    for tok, cnt in freq.most_common():
        if tok in stopwords or tok in domain_stop:
            continue
        if len(tok) < 3:
            continue
        keywords.append(tok)
        if len(keywords) >= top_n:
            break
    return keywords


def visualize_clusters(X_2d, labels, output_path, epic_names=None):
    """
    Create a 2D visualization of clusters using matplotlib.
    X_2d should be a 2D array (e.g., from UMAP or t-SNE).
    """
    if not HAS_VIZ:
        print("[WARN] Visualization libraries not available, skipping plots.")
        return
    
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = sns.color_palette('husl', n_colors=len(unique_labels))
    
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # Outliers in HDBSCAN
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c='gray', marker='x', alpha=0.5, label='Noise')
        else:
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=[colors[idx]], label=f'Cluster {label}', 
                       alpha=0.6, edgecolors='w', linewidth=0.5)
    
    plt.title('Issue Clusters Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved cluster visualization to: {output_path}")
    plt.close()


def cluster_with_hdbscan(X, min_cluster_size=3, min_samples=2):
    """
    Cluster using HDBSCAN (density-based clustering).
    Automatically determines the number of clusters and handles outliers.
    """
    if not HAS_HDBSCAN:
        raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
    
    print(f"[INFO] Clustering with HDBSCAN (min_cluster_size={min_cluster_size})")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[INFO] HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
    
    return labels


def cluster_with_bertopic(texts, min_topic_size=3):
    """
    Use BERTopic for topic modeling, which combines embeddings with clustering.
    Returns topics (cluster labels) and the BERTopic model.
    """
    if not HAS_BERTOPIC:
        raise ImportError("BERTopic not available. Install with: pip install bertopic")
    
    print(f"[INFO] Running BERTopic topic modeling (min_topic_size={min_topic_size})")
    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        verbose=True,
        calculate_probabilities=False
    )
    topics, probs = topic_model.fit_transform(texts)
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f"[INFO] BERTopic found {n_topics} topics")
    
    return topics, topic_model


def cluster_and_create_epics(
    df: pd.DataFrame,
    summary_col: str,
    description_col: str,
    issue_type_col: str,
    min_issues_per_epic: int = 3,
    user_k: int = None,
    model_name: str = "all-MiniLM-L6-v2",
    method: str = "hdbscan",
    use_umap: bool = True,
    optimize_k: bool = False,
    output_dir: str = None,
    batch_size: int = None,
):
    """
    Cluster non-Epic issues into thematic groups and derive Epics.
    Returns:
        df_with_epics: original df plus columns [Suggested Epic ID, Suggested Epic Name]
        epics_df: dataframe describing each proposed Epic
    """
    df = df.copy()

    # Filter out existing Epics (we only cluster Stories/Tasks/Bugs/etc.)
    issue_type_lower = df[issue_type_col].astype(str).str.lower()
    non_epic_mask = issue_type_lower != "epic"
    df_non_epic = df[non_epic_mask].copy()
    print(f"[INFO] Non-epic issues to cluster: {len(df_non_epic)}")

    if df_non_epic.empty:
        print("[WARN] No non-epic issues found. Nothing to cluster.")
        df["Suggested Epic ID"] = np.nan
        df["Suggested Epic Name"] = np.nan
        epics_df = pd.DataFrame(columns=[
            "Epic ID", "Epic Name", "Cluster ID",
            "Issue Count", "Theme Keywords", "Representative Summaries"
        ])
        return df, epics_df

    texts = build_text_series(df_non_epic, summary_col, description_col)

    # Handle different clustering methods
    topic_model = None
    
    if method == "bertopic":
        # BERTopic handles embeddings internally
        if not HAS_BERTOPIC:
            print("[ERROR] BERTopic not available. Falling back to HDBSCAN.")
            method = "hdbscan"
        else:
            labels, topic_model = cluster_with_bertopic(
                texts.tolist(), 
                min_topic_size=min_issues_per_epic
            )
            df_non_epic["cluster"] = labels
            X = None  # Not needed for visualization in this path
    
    if method != "bertopic":
        # Compute embeddings for HDBSCAN/KMeans methods
        X, embedding_type = compute_embeddings(texts, model_name=model_name, batch_size=batch_size)
        print(f"[INFO] Embedding type used: {embedding_type}")
        
        # Apply UMAP dimensionality reduction if requested
        X_clustered = X
        if use_umap and HAS_UMAP:
            X_clustered = reduce_dimensions_umap(X, n_components=5)
        elif use_umap:
            print("[WARN] UMAP not available, skipping dimensionality reduction.")
        
        # Cluster based on method
        if method == "hdbscan":
            if not HAS_HDBSCAN:
                print("[ERROR] HDBSCAN not available. Falling back to KMeans.")
                method = "kmeans"
            else:
                labels = cluster_with_hdbscan(
                    X_clustered, 
                    min_cluster_size=min_issues_per_epic
                )
                df_non_epic["cluster"] = labels
        
        if method == "kmeans":
            # Determine k
            if optimize_k and not user_k:
                k = find_optimal_clusters_silhouette(X_clustered)
            else:
                k = choose_cluster_count(len(df_non_epic), user_k=user_k)
            
            print(f"[INFO] Clustering into k={k} clusters using KMeans.")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_clustered)
            df_non_epic["cluster"] = labels
        
        # Create visualization if output directory provided
        if output_dir and HAS_VIZ:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Reduce to 2D for visualization
            if HAS_UMAP:
                print("[INFO] Creating 2D UMAP projection for visualization...")
                X_2d = reduce_dimensions_umap(X, n_components=2)
                viz_path = output_dir / "clusters_umap.png"
                visualize_clusters(X_2d, labels, viz_path)
            
            # Also create t-SNE visualization
            print("[INFO] Creating t-SNE projection for visualization...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_tsne = tsne.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
            viz_path = output_dir / "clusters_tsne.png"
            visualize_clusters(X_tsne, labels, viz_path)

    # Prepare columns for suggestions
    df["Suggested Epic ID"] = np.nan
    df["Suggested Epic Name"] = np.nan

    epic_meta = []
    epic_id_counter = 1

    for clus in sorted(df_non_epic["cluster"].unique()):
        # Skip noise cluster in HDBSCAN
        if clus == -1:
            print(f"[INFO] Skipping noise cluster (outliers)")
            continue
            
        cluster_idx = df_non_epic.index[df_non_epic["cluster"] == clus]
        size = len(cluster_idx)
        if size < min_issues_per_epic:
            # Too small to promote as a proper Epic
            continue

        # Derive a name from keywords
        cluster_texts = (
            df_non_epic.loc[cluster_idx, summary_col]
            .fillna("")
            .astype(str)
            .tolist()
        )
        
        # Use BERTopic topic labels if available
        if topic_model is not None:
            try:
                topic_info = topic_model.get_topic(clus)
                if topic_info:
                    # Get top words from topic
                    kws = [word for word, score in topic_info[:4]]
                    epic_name = "Epic: " + " / ".join(kws[:3])
                else:
                    kws = extract_keywords(cluster_texts, top_n=4)
                    epic_name = "Epic: " + " / ".join(kws[:3]) if kws else f"Epic {epic_id_counter}"
            except:
                kws = extract_keywords(cluster_texts, top_n=4)
                epic_name = "Epic: " + " / ".join(kws[:3]) if kws else f"Epic {epic_id_counter}"
        else:
            # Use KeyBERT or frequency-based extraction
            kws = extract_keywords(cluster_texts, top_n=4, use_keybert=True)
            if kws:
                epic_name = "Epic: " + " / ".join(kws[:3])
            else:
                epic_name = f"Epic {epic_id_counter}"

        epic_id = f"E{epic_id_counter}"
        epic_id_counter += 1

        print(f"[INFO] Created {epic_id} -> {epic_name} (size={size})")

        # Assign suggested epic to all items in this cluster
        for idx in cluster_idx:
            df.at[idx, "Suggested Epic ID"] = epic_id
            df.at[idx, "Suggested Epic Name"] = epic_name

        # Capture metadata for the Epics sheet
        sample_summaries = df_non_epic.loc[cluster_idx, summary_col].head(3).tolist()
        epic_meta.append({
            "Epic ID": epic_id,
            "Epic Name": epic_name,
            "Cluster ID": int(clus),
            "Issue Count": int(size),
            "Theme Keywords": ", ".join(kws),
            "Representative Summaries": "; ".join(sample_summaries),
        })

    epics_df = pd.DataFrame(epic_meta)
    return df, epics_df


def main():
    parser = argparse.ArgumentParser(
        description="Create suggested Epics for Jira issues using ML clustering."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to Jira export (XLSX)."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output XLSX with suggested Epics."
    )
    parser.add_argument(
        "--sheet-name", default=None,
        help="Name of sheet containing issues (default: first sheet)."
    )
    parser.add_argument(
        "--model-name", default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name to use. "
             "Lightweight: all-MiniLM-L6-v2 (80M params, fast), "
             "Standard: all-mpnet-base-v2 (110M params, balanced), "
             "paraphrase-multilingual-mpnet-base-v2 (278M params, multilingual). "
             "Heavy-duty (GPU recommended): sentence-transformers/all-roberta-large-v1 (355M params), "
             "BAAI/bge-large-en-v1.5 (335M params, SOTA), "
             "intfloat/e5-large-v2 (335M params), "
             "sentence-transformers/gtr-t5-xl (1.2B params, best quality). "
             "Instruction-tuned: hkunlp/instructor-xl (1.5B params, task-specific)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for encoding. Auto-detected based on hardware. "
             "CPU: 8, Apple Silicon: 16, NVIDIA GPU: 32-64. "
             "Reduce if out of memory."
    )
    parser.add_argument(
        "--method", default="hdbscan",
        choices=["hdbscan", "kmeans", "bertopic"],
        help="Clustering method: hdbscan (density-based, auto clusters), "
             "kmeans (traditional), bertopic (topic modeling)."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Optional fixed number of clusters for KMeans (default: auto)."
    )
    parser.add_argument(
        "--optimize-k", action="store_true",
        help="Use silhouette analysis to find optimal k for KMeans (slower)."
    )
    parser.add_argument(
        "--use-umap", action="store_true", default=True,
        help="Use UMAP dimensionality reduction before clustering (recommended)."
    )
    parser.add_argument(
        "--no-umap", action="store_false", dest="use_umap",
        help="Disable UMAP dimensionality reduction."
    )
    parser.add_argument(
        "--min-issues-per-epic", type=int, default=3,
        help="Minimum issue count in a cluster to promote it as an Epic."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate cluster visualization plots (saved alongside output)."
    )
    args = parser.parse_args()

    # Display hardware capabilities
    print("[INFO] ===== Hardware Detection =====")
    if HAS_SENTENCE_TRANSFORMERS:
        if HAS_CUDA:
            print(f"[INFO] ✓ CUDA GPU available (NVIDIA)")
            import torch
            print(f"[INFO]   GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO]   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif HAS_MPS:
            print(f"[INFO] ✓ MPS GPU available (Apple Silicon)")
        else:
            print(f"[INFO] ⚠ No GPU detected, using CPU (slower for large models)")
    print(f"[INFO] ================================\n")

    print(f"[INFO] Loading Excel file: {args.input}")
    xls = pd.ExcelFile(args.input)
    if args.sheet_name is not None:
        sheet_name = args.sheet_name
    else:
        sheet_name = xls.sheet_names[0]
    df = pd.read_excel(args.input, sheet_name=sheet_name)
    print(f"[INFO] Using sheet: {sheet_name}")
    print(f"[INFO] Total rows: {len(df)}")

    # Identify important columns
    summary_col = find_column(df, ["summary"])
    description_col = find_column(df, ["description"])
    issue_type_col = find_column(df, ["issue type", "issuetype"])
    key_col = find_column(df, ["issue key", "key"])

    print(f"[INFO] Detected columns:")
    print(f"       Summary:      {summary_col}")
    print(f"       Description:  {description_col}")
    print(f"       Issue Type:   {issue_type_col}")
    print(f"       Issue Key:    {key_col}")

    required = {
        "Summary": summary_col,
        "Description": description_col,
        "Issue Type": issue_type_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Could not find the following required columns: {', '.join(missing)}. "
            "Please rename your columns or adjust find_column()."
        )

    # Determine output directory for visualizations
    output_dir = None
    if args.visualize:
        output_dir = Path(args.output).parent / "cluster_visualizations"
        print(f"[INFO] Visualizations will be saved to: {output_dir}")
    
    # Cluster and create epics
    df_with_epics, epics_df = cluster_and_create_epics(
        df,
        summary_col=summary_col,
        description_col=description_col,
        issue_type_col=issue_type_col,
        min_issues_per_epic=args.min_issues_per_epic,
        user_k=args.k,
        model_name=args.model_name,
        method=args.method,
        use_umap=args.use_umap,
        optimize_k=args.optimize_k,
        output_dir=output_dir,
        batch_size=args.batch_size,
    )

    # Write output Excel
    print(f"[INFO] Writing output to: {args.output}")
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        df_with_epics.to_excel(writer, sheet_name="Issues_Updated", index=False)
        epics_df.to_excel(writer, sheet_name="Epics", index=False)

    print("[INFO] Done. Review 'Issues_Updated' and 'Epics' sheets in the output file.")


if __name__ == "__main__":
    main()
