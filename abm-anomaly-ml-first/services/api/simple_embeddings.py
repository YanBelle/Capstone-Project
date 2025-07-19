#!/usr/bin/env python3
"""
Simple embedding generator that doesn't rely on huggingface_hub
Uses TF-IDF as a fallback for embeddings if transformers fail
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)

class SimpleEmbeddingGenerator:
    """Fallback embedding generator using TF-IDF and SVD"""
    
    def __init__(self, n_components=384):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """Fit the model and transform texts to embeddings"""
        logger.info("Generating TF-IDF based embeddings")
        
        # Convert to TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Reduce dimensionality with SVD
        embeddings = self.svd.fit_transform(tfidf_matrix)
        
        self.is_fitted = True
        logger.info(f"Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")
        
        return embeddings
    
    def transform(self, texts):
        """Transform texts to embeddings using fitted model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf_matrix)
        
        return embeddings

def get_simple_embeddings(texts):
    """Simple function to get embeddings without external dependencies"""
    generator = SimpleEmbeddingGenerator()
    return generator.fit_transform(texts)
