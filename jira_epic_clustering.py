"""
jira_epic_clustering.py

Use advanced ML (SentenceTransformers + clustering) to automatically propose Epics
for Jira issues based on their Summary + Description.

Usage:
    python jira_epic_clustering.py \
        --input Jira_Export.xlsx \
        --output Jira_Export_with_epics.xlsx \
        --cluster-method auto \
        --k-max 12

Requirements (install via pip as needed):
    - pandas
    - numpy
    - scikit-learn
    - sentence-transformers  (preferred, for semantic embeddings)
    - hdbscan (optional, for density-based clustering)
    - umap-learn (optional, improves HDBSCAN performance)
    - keybert (optional, higher-quality keywords for epic names)
    
If sentence-transformers is not available, the script will fall back to a
TF-IDF + KMeans pipeline (still using scikit-learn).
"""

import argparse
import math
import re
import warnings
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

import scipy.sparse as sp

# Try to import SentenceTransformer (advanced semantic embeddings)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

# Optional: density-based clustering for automatic cluster discovery
try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# Optional: dimensionality reduction to improve HDBSCAN separation
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional: better keyword extraction for naming epics
try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except Exception:
    HAS_KEYBERT = False


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


def compute_embeddings(
    texts,
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, str, Optional[object]]:
    """
    Compute sentence embeddings for the given texts using SentenceTransformers.
    Returns (embeddings, embedding_type, embedding_model).
    Falls back to TF-IDF embeddings if sentence-transformers is not available.
    The returned embedding_model can be reused for KeyBERT keyword extraction.
    """
    if HAS_SENTENCE_TRANSFORMERS:
        print(f"[INFO] Using SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts.tolist(), show_progress_bar=True)
        return np.array(embeddings), "sentence-transformers", model
    else:
        print("[WARN] sentence-transformers not available. Falling back to TF-IDF.")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2000,
        )
        X = vectorizer.fit_transform(texts)
        return X, "tfidf", None


def perform_kmeans(
    X,
    user_k: Optional[int] = None,
    k_max: int = 12,
    random_state: int = 42,
):
    """
    Run KMeans, optionally searching over multiple k using silhouette score.
    Returns (labels, model, k_used, silhouette_score_or_None).
    """
    n_items = X.shape[0]
    if user_k is not None and user_k > 1:
        candidate_ks = [int(user_k)]
    else:
        upper = min(k_max, max(3, int(round(math.sqrt(max(n_items, 2)) * 1.5))))
        candidate_ks = list(range(2, upper + 1))

    best_model = None
    best_labels = None
    best_score = -np.inf
    best_k = None

    for k in candidate_ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)

        if len(set(labels)) < 2:
            score = -np.inf
        else:
            try:
                score = silhouette_score(X, labels)
            except Exception as exc:  # sklearn can fail on degenerate inputs
                warnings.warn(f"Silhouette failed for k={k}: {exc}")
                score = -np.inf

        if score > best_score:
            best_model = model
            best_labels = labels
            best_score = score
            best_k = k

    if best_model is None:
        # Fallback: just run with smallest candidate k
        fallback_k = candidate_ks[0]
        best_model = KMeans(n_clusters=fallback_k, random_state=random_state, n_init=10)
        best_labels = best_model.fit_predict(X)
        best_k = fallback_k
        best_score = None

    return best_labels, best_model, best_k, (None if best_score == -np.inf else best_score)


def try_hdbscan(
    X,
    min_cluster_size: int,
    embedding_type: str,
):
    """
    Run HDBSCAN clustering if the library is available and embeddings are dense.
    Returns (labels, info_dict), or (None, {}) on failure.
    """
    if not HAS_HDBSCAN:
        return None, {}

    if sp.issparse(X):
        warnings.warn("[INFO] HDBSCAN skipped because embeddings are sparse (TF-IDF).")
        return None, {}

    X_for_clustering = X
    reducer = None

    # Optional UMAP pre-step to make HDBSCAN more stable on high dimensions
    if HAS_UMAP and X.shape[1] > 50:
        try:
            reducer = umap.UMAP(
                n_components=15,
                n_neighbors=15,
                random_state=42,
                metric="cosine" if embedding_type == "sentence-transformers" else "euclidean",
            )
            X_for_clustering = reducer.fit_transform(X)
            print("[INFO] Applied UMAP dimensionality reduction before HDBSCAN.")
        except Exception as exc:
            warnings.warn(f"UMAP preprocessing failed; continuing without it. Error: {exc}")
            X_for_clustering = X

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=max(1, min_cluster_size // 2),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(X_for_clustering)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        info = {
            "method": "hdbscan",
            "n_clusters": n_clusters,
            "noise_points": int(np.sum(labels == -1)),
            "probabilities": clusterer.probabilities_,
        }
        if reducer is not None:
            info["umap_reducer"] = reducer
        if n_clusters == 0:
            warnings.warn("HDBSCAN found only noise; consider lowering min_cluster_size or using KMeans.")
        return labels, info
    except Exception as exc:
        warnings.warn(f"HDBSCAN failed, will fall back to KMeans. Error: {exc}")
        return None, {}


def cluster_embeddings(
    X,
    embedding_type: str,
    method: str = "auto",
    user_k: Optional[int] = None,
    k_max: int = 12,
    min_cluster_size: int = 3,
):
    """
    Cluster embeddings using the requested method with sensible fallbacks.
    Returns (labels, clustering_info_dict).
    """
    method = method.lower()

    # Try density-based clustering first if requested/allowed
    if method in {"hdbscan", "auto"} and embedding_type == "sentence-transformers":
        labels, info = try_hdbscan(X, min_cluster_size=min_cluster_size, embedding_type=embedding_type)
        if labels is not None and info.get("n_clusters", 0) > 0:
            print(f"[INFO] Using HDBSCAN (clusters={info['n_clusters']}, noise={info['noise_points']}).")
            return labels, info
        if method == "hdbscan":
            print("[WARN] HDBSCAN not available or failed; falling back to KMeans.")

    # Fallback: KMeans with silhoutte-guided k selection
    labels, model, k_used, sil_score = perform_kmeans(
        X, user_k=user_k, k_max=k_max, random_state=42
    )
    info = {
        "method": "kmeans",
        "k": k_used,
        "silhouette": sil_score,
        "model": model,
    }
    score_txt = f"{sil_score:.3f}" if sil_score is not None else "n/a"
    print(f"[INFO] Using KMeans with k={k_used} (silhouette={score_txt}).")
    return labels, info


def extract_keywords(texts, top_n=5, keybert_model=None):
    """
    Extract keywords from a list of text snippets, preferring KeyBERT if available
    (for multi-word, semantically-aware phrases) and falling back to frequency.
    """
    stopwords = set(ENGLISH_STOP_WORDS)
    domain_stop = {
        "lrt", "jira", "request", "requests", "story", "task",
        "bug", "issue", "sprint", "epic", "project"
    }

    combined = " ".join(texts).strip()

    if keybert_model is not None and combined:
        try:
            # Use MMR for diverse yet relevant keywords
            candidates = keybert_model.extract_keywords(
                combined,
                top_n=top_n * 2,
                stop_words="english",
                use_mmr=True,
                diversity=0.6,
            )
            cleaned = []
            for phrase, _score in candidates:
                token_norm = phrase.lower()
                token_parts = re.findall(r"\b[a-zA-Z0-9]+\b", token_norm)
                if any((part in stopwords or part in domain_stop) for part in token_parts):
                    continue
                if len(token_norm) < 3:
                    continue
                cleaned.append(phrase)
                if len(cleaned) >= top_n:
                    break
            if cleaned:
                return cleaned
        except Exception as exc:
            warnings.warn(f"KeyBERT keyword extraction failed; using frequency. Error: {exc}")

    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]+\b", combined.lower())
    freq = Counter(tokens)
    keywords = []
    for tok, _cnt in freq.most_common():
        if tok in stopwords or tok in domain_stop:
            continue
        if len(tok) < 3:
            continue
        keywords.append(tok)
        if len(keywords) >= top_n:
            break
    return keywords


def cluster_and_create_epics(
    df: pd.DataFrame,
    summary_col: str,
    description_col: str,
    issue_type_col: str,
    min_issues_per_epic: int = 3,
    user_k: int = None,
    model_name: str = "all-MiniLM-L6-v2",
    cluster_method: str = "auto",
    k_max: int = 12,
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
            "Issue Count", "Cluster Confidence",
            "Theme Keywords", "Representative Summaries",
            "Clustering Method",
        ])
        return df, epics_df

    texts = build_text_series(df_non_epic, summary_col, description_col)

    # Compute embeddings
    X, embedding_type, embedding_model = compute_embeddings(texts, model_name=model_name)
    print(f"[INFO] Embedding type used: {embedding_type}")

    # Optional: KeyBERT for better epic naming
    keybert_model = None
    if HAS_KEYBERT and embedding_model is not None:
        try:
            keybert_model = KeyBERT(model=embedding_model)
            print("[INFO] Using KeyBERT for epic keywording.")
        except Exception as exc:
            warnings.warn(f"KeyBERT initialization failed; using frequency keywords. Error: {exc}")

    # Cluster
    labels, clustering_info = cluster_embeddings(
        X,
        embedding_type=embedding_type,
        method=cluster_method,
        user_k=user_k,
        k_max=k_max,
        min_cluster_size=min_issues_per_epic,
    )
    df_non_epic["cluster"] = labels

    # Compute simple confidence per cluster (HDBSCAN: mean probability; KMeans: silhouette)
    cluster_confidence = {}
    unique_clusters = [c for c in sorted(df_non_epic["cluster"].unique()) if c != -1]
    if clustering_info.get("method") == "hdbscan":
        probs = clustering_info.get("probabilities")
        if probs is not None:
            for clus in unique_clusters:
                mask = (df_non_epic["cluster"] == clus).to_numpy()
                cluster_confidence[clus] = float(np.mean(probs[mask]))
    else:
        sil = clustering_info.get("silhouette")
        for clus in unique_clusters:
            cluster_confidence[clus] = sil if sil is not None else np.nan

    # Prepare columns for suggestions
    df["Suggested Epic ID"] = np.nan
    df["Suggested Epic Name"] = np.nan

    epic_meta = []
    epic_id_counter = 1

    if not unique_clusters:
        print("[WARN] No valid clusters found; skipping epic suggestions.")

    for clus in unique_clusters:
        cluster_idx = df_non_epic.index[df_non_epic["cluster"] == clus]
        size = len(cluster_idx)
        if size < min_issues_per_epic:
            # Too small to promote as a proper Epic
            continue

        # Derive a name from the most frequent keywords
        cluster_texts = (
            df_non_epic.loc[cluster_idx, summary_col]
            .fillna("")
            .astype(str)
            .tolist()
        )
        kws = extract_keywords(cluster_texts, top_n=4, keybert_model=keybert_model)
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
            "Cluster Confidence": cluster_confidence.get(clus),
            "Theme Keywords": ", ".join(kws),
            "Representative Summaries": "; ".join(sample_summaries),
            "Clustering Method": clustering_info.get("method"),
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
        help="SentenceTransformer model name to use (if installed)."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Optional fixed number of clusters for KMeans (default: auto-selection)."
    )
    parser.add_argument(
        "--k-max", type=int, default=12,
        help="Maximum clusters to consider when auto-selecting k for KMeans."
    )
    parser.add_argument(
        "--cluster-method", default="auto", choices=["auto", "kmeans", "hdbscan"],
        help="Clustering algorithm: auto prefers HDBSCAN on dense embeddings and falls back to KMeans."
    )
    parser.add_argument(
        "--min-issues-per-epic", type=int, default=3,
        help="Minimum issue count in a cluster to promote it as an Epic."
    )
    args = parser.parse_args()

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

    # Cluster and create epics
    df_with_epics, epics_df = cluster_and_create_epics(
        df,
        summary_col=summary_col,
        description_col=description_col,
        issue_type_col=issue_type_col,
        min_issues_per_epic=args.min_issues_per_epic,
        user_k=args.k,
        model_name=args.model_name,
        cluster_method=args.cluster_method,
        k_max=args.k_max,
    )

    # Write output Excel
    print(f"[INFO] Writing output to: {args.output}")
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        df_with_epics.to_excel(writer, sheet_name="Issues_Updated", index=False)
        epics_df.to_excel(writer, sheet_name="Epics", index=False)

    print("[INFO] Done. Review 'Issues_Updated' and 'Epics' sheets in the output file.")


if __name__ == "__main__":
    main()
