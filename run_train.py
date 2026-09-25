#!/usr/bin/env python3
"""Train the entity resolution model.

Usage:
    python run_train.py
"""
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import train_pipeline

if __name__ == '__main__':
    train_pipeline()
