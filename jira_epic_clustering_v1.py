"""
jira_epic_clustering.py

Use advanced ML (SentenceTransformers + clustering) to automatically propose Epics
for Jira issues based on their Summary + Description.

Usage:
    python jira_epic_clustering.py \
        --input Jira_Export.xlsx \
        --output Jira_Export_with_epics.xlsx

Requirements (install via pip as needed):
    - pandas
    - numpy
    - scikit-learn
    - sentence-transformers  (preferred, for semantic embeddings)
    
If sentence-transformers is not available, the script will fall back to a
TF-IDF + KMeans pipeline (still using scikit-learn).
"""

import argparse
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Try to import SentenceTransformer (advanced semantic embeddings)
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


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


def compute_embeddings(texts, model_name: str = "all-MiniLM-L6-v2"):
    """
    Compute sentence embeddings for the given texts using SentenceTransformers.
    Falls back to TF-IDF embeddings if sentence-transformers is not available.
    """
    if HAS_SENTENCE_TRANSFORMERS:
        print(f"[INFO] Using SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts.tolist(), show_progress_bar=True)
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


def extract_keywords(texts, top_n=5):
    """
    Extract simple frequency-based keywords from a list of text snippets,
    ignoring common stopwords and domain-generic words.
    """
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


def cluster_and_create_epics(
    df: pd.DataFrame,
    summary_col: str,
    description_col: str,
    issue_type_col: str,
    min_issues_per_epic: int = 3,
    user_k: int = None,
    model_name: str = "all-MiniLM-L6-v2",
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

    # Compute embeddings
    X, embedding_type = compute_embeddings(texts, model_name=model_name)
    print(f"[INFO] Embedding type used: {embedding_type}")

    # Cluster
    k = choose_cluster_count(len(df_non_epic), user_k=user_k)
    print(f"[INFO] Clustering into k={k} clusters using KMeans.")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    df_non_epic["cluster"] = labels

    # Prepare columns for suggestions
    df["Suggested Epic ID"] = np.nan
    df["Suggested Epic Name"] = np.nan

    epic_meta = []
    epic_id_counter = 1

    for clus in sorted(df_non_epic["cluster"].unique()):
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
        kws = extract_keywords(cluster_texts, top_n=4)
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
        help="SentenceTransformer model name to use (if installed)."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Optional fixed number of clusters for KMeans (default: auto)."
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
    )

    # Write output Excel
    print(f"[INFO] Writing output to: {args.output}")
    with pd.ExcelWriter(args.output, engine="xlsxwriter") as writer:
        df_with_epics.to_excel(writer, sheet_name="Issues_Updated", index=False)
        epics_df.to_excel(writer, sheet_name="Epics", index=False)

    print("[INFO] Done. Review 'Issues_Updated' and 'Epics' sheets in the output file.")


if __name__ == "__main__":
    main()
