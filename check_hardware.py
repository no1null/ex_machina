#!/usr/bin/env python3
"""
Hardware detection and model recommendation tool for Jira Epic Clustering.

This script detects available hardware (CPU, GPU) and recommends optimal
models and settings for the clustering script.
"""

import sys

def check_hardware():
    """Detect hardware and recommend models."""
    print("=" * 70)
    print("HARDWARE DETECTION & MODEL RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Check PyTorch and GPU availability
    try:
        import torch
        print("✓ PyTorch installed")
        print(f"  Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("\n✓ NVIDIA GPU (CUDA) DETECTED")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print("\n" + "=" * 70)
            print("RECOMMENDED MODELS FOR YOUR HARDWARE:")
            print("=" * 70)
            
            if vram_gb >= 16:
                print("\n🚀 HEAVY-DUTY MODELS (Best Quality)")
                print("   Your GPU can handle the largest models!")
                print()
                print("   1. hkunlp/instructor-xl (1.5B params)")
                print("      - Task-specific instructions for better results")
                print("      - Batch size: 8-16")
                print()
                print("   2. sentence-transformers/gtr-t5-xl (1.2B params)")
                print("      - Best overall quality")
                print("      - Batch size: 8-16")
                print()
                print("   3. BAAI/bge-large-en-v1.5 (335M params)")
                print("      - State-of-the-art, very fast")
                print("      - Batch size: 32-64")
            
            elif vram_gb >= 8:
                print("\n💪 LARGE MODELS (Excellent Quality)")
                print()
                print("   1. BAAI/bge-large-en-v1.5 (335M params) [RECOMMENDED]")
                print("      - State-of-the-art performance")
                print("      - Batch size: 32-48")
                print()
                print("   2. intfloat/e5-large-v2 (335M params)")
                print("      - Very strong performance")
                print("      - Batch size: 32-48")
                print()
                print("   3. sentence-transformers/all-roberta-large-v1 (355M params)")
                print("      - Robust, well-tested")
                print("      - Batch size: 24-32")
            
            else:
                print("\n⚡ MEDIUM MODELS (Good Quality)")
                print()
                print("   1. BAAI/bge-base-en-v1.5 (110M params) [RECOMMENDED]")
                print("      - Excellent quality-to-speed ratio")
                print("      - Batch size: 64-128")
                print()
                print("   2. all-mpnet-base-v2 (110M params)")
                print("      - Widely used, reliable")
                print("      - Batch size: 64-128")
            
        elif torch.backends.mps.is_available():
            print("\n✓ APPLE SILICON (MPS) DETECTED")
            print("  Metal Performance Shaders enabled")
            
            print("\n" + "=" * 70)
            print("RECOMMENDED MODELS FOR APPLE SILICON:")
            print("=" * 70)
            print()
            print("💻 OPTIMIZED FOR APPLE SILICON")
            print()
            print("   1. BAAI/bge-large-en-v1.5 (335M params) [RECOMMENDED]")
            print("      - Excellent performance on M-series chips")
            print("      - Batch size: 16-32")
            print()
            print("   2. all-mpnet-base-v2 (110M params)")
            print("      - Fast and efficient")
            print("      - Batch size: 32-64")
            print()
            print("   3. intfloat/e5-large-v2 (335M params)")
            print("      - High quality results")
            print("      - Batch size: 16-24")
        
        else:
            print("\nℹ CPU ONLY (No GPU detected)")
            
            print("\n" + "=" * 70)
            print("RECOMMENDED MODELS FOR CPU:")
            print("=" * 70)
            print()
            print("⚡ LIGHTWEIGHT MODELS (CPU-Optimized)")
            print()
            print("   1. all-MiniLM-L6-v2 (80M params) [DEFAULT]")
            print("      - Very fast on CPU")
            print("      - Batch size: 8-16")
            print()
            print("   2. all-MiniLM-L12-v2 (33M params)")
            print("      - Smaller, even faster")
            print("      - Batch size: 16-32")
            print()
            print("   💡 Consider using a GPU instance for large models")
    
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Install with: pip install torch")
        return
    
    # Check sentence-transformers
    print("\n" + "=" * 70)
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers installed")
    except ImportError:
        print("✗ sentence-transformers not installed")
        print("  Install with: pip install sentence-transformers")
    
    # Check other libraries
    try:
        import hdbscan
        print("✓ hdbscan installed")
    except ImportError:
        print("✗ hdbscan not installed")
    
    try:
        import umap
        print("✓ umap-learn installed")
    except ImportError:
        print("✗ umap-learn not installed")
    
    try:
        from bertopic import BERTopic
        print("✓ bertopic installed")
    except ImportError:
        print("✗ bertopic not installed")
    
    try:
        from keybert import KeyBERT
        print("✓ keybert installed")
    except ImportError:
        print("✗ keybert not installed")
    
    print("\n" + "=" * 70)
    print("EXAMPLE COMMANDS:")
    print("=" * 70)
    print()
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 16:
            print("# Heavy-duty model (best quality):")
            print("python jira_epic_clustering_v1.py \\")
            print("    --input data.xlsx \\")
            print("    --output output.xlsx \\")
            print("    --method hdbscan \\")
            print("    --model-name sentence-transformers/gtr-t5-xl \\")
            print("    --batch-size 12 \\")
            print("    --visualize")
        else:
            print("# Large model (excellent quality):")
            print("python jira_epic_clustering_v1.py \\")
            print("    --input data.xlsx \\")
            print("    --output output.xlsx \\")
            print("    --method hdbscan \\")
            print("    --model-name BAAI/bge-large-en-v1.5 \\")
            print("    --batch-size 32 \\")
            print("    --visualize")
    elif torch.backends.mps.is_available():
        print("# Apple Silicon optimized:")
        print("python jira_epic_clustering_v1.py \\")
        print("    --input data.xlsx \\")
        print("    --output output.xlsx \\")
        print("    --method hdbscan \\")
        print("    --model-name BAAI/bge-large-en-v1.5 \\")
        print("    --batch-size 16 \\")
        print("    --visualize")
    else:
        print("# CPU optimized (lightweight):")
        print("python jira_epic_clustering_v1.py \\")
        print("    --input data.xlsx \\")
        print("    --output output.xlsx \\")
        print("    --method hdbscan \\")
        print("    --model-name all-MiniLM-L6-v2 \\")
        print("    --batch-size 8 \\")
        print("    --visualize")
    
    print()
    print("=" * 70)
    print()


if __name__ == "__main__":
    check_hardware()
