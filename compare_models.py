#!/usr/bin/env python3
"""
Quick model comparison script.
Tests different models on a small sample to show quality differences.
"""

import sys
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Please install required packages:")
    print("pip install sentence-transformers numpy scikit-learn")
    sys.exit(1)

# Sample Jira issue summaries for testing
SAMPLE_ISSUES = [
    # Group 1: Authentication/Login issues
    "User cannot login to the system",
    "Login page shows 500 error",
    "Password reset functionality not working",
    
    # Group 2: Performance issues
    "Dashboard loads very slowly",
    "Query timeout when generating reports",
    "Page freezes when loading large datasets",
    
    # Group 3: UI/UX issues
    "Button alignment issue on mobile",
    "Dropdown menu not showing all options",
    "Inconsistent spacing between elements",
]

def test_model(model_name, batch_size=8):
    """Test a model's semantic understanding."""
    print(f"\nTesting: {model_name}")
    print("-" * 70)
    
    try:
        model = SentenceTransformer(model_name)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        
        # Encode
        embeddings = model.encode(SAMPLE_ISSUES, batch_size=batch_size, show_progress_bar=False)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        # Show how well it groups similar issues
        print("\nSemantic similarity examples:")
        print(f"  Login vs Password Reset: {sim_matrix[0, 2]:.3f}")
        print(f"  Login vs Dashboard Load: {sim_matrix[0, 3]:.3f}")
        print(f"  Dashboard vs Query Timeout: {sim_matrix[3, 4]:.3f}")
        print(f"  Dashboard vs UI Button: {sim_matrix[3, 6]:.3f}")
        
        # Calculate average within-group vs between-group similarity
        auth_group = [0, 1, 2]
        perf_group = [3, 4, 5]
        ui_group = [6, 7, 8]
        
        within_auth = np.mean([sim_matrix[i, j] for i in auth_group for j in auth_group if i < j])
        within_perf = np.mean([sim_matrix[i, j] for i in perf_group for j in perf_group if i < j])
        within_ui = np.mean([sim_matrix[i, j] for i in ui_group for j in ui_group if i < j])
        
        between_groups = np.mean([sim_matrix[i, j] 
                                  for i in auth_group 
                                  for j in perf_group + ui_group])
        
        print(f"\nAvg within-group similarity: {np.mean([within_auth, within_perf, within_ui]):.3f}")
        print(f"Avg between-group similarity: {between_groups:.3f}")
        print(f"Separation ratio: {np.mean([within_auth, within_perf, within_ui]) / between_groups:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 70)
    print("MODEL COMPARISON TEST")
    print("=" * 70)
    print("\nThis will test semantic understanding of different models")
    print("Higher 'separation ratio' = better clustering quality\n")
    
    models_to_test = [
        ("all-MiniLM-L6-v2", 8, "Lightweight (Default)"),
        ("all-mpnet-base-v2", 8, "Medium"),
        ("BAAI/bge-large-en-v1.5", 8, "Large (Recommended)"),
    ]
    
    for model_name, batch_size, description in models_to_test:
        print(f"\n{'=' * 70}")
        print(f"[{description}]")
        test_model(model_name, batch_size)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nLarger models generally show:")
    print("  ✓ Higher within-group similarity")
    print("  ✓ Lower between-group similarity")
    print("  ✓ Better separation ratio")
    print("  ✓ More accurate Epic clustering")
    print("\nRecommendation: BAAI/bge-large-en-v1.5 for best results")
    print("=" * 70)

if __name__ == "__main__":
    main()
