#!/usr/bin/env python3
"""Generate test predictions using the trained model.

Usage:
    python run_predict.py

Outputs:
    output/matching_results.tsv   — scored on the leaderboard
    output/candidate_pairs.tsv    — blocking audit file
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import predict_pipeline

if __name__ == '__main__':
    predict_pipeline()
