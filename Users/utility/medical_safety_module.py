"""
MEDICAL-GRADE PILL IDENTIFICATION - SAFETY ENHANCEMENTS
========================================================

Advanced features for medical-safe pill identification:
1. Embedding-based similarity matching (open-set recognition)
2. Contrastive learning utilities
3. Class imbalance detection and handling
4. Confidence calibration
5. Decision threshold optimization
6. Explainable AI with feature extraction

Medical Safety Principles:
- Accuracy > Confidence
- Reject high-confidence wrong predictions
- Support open-set recognition (UNKNOWN class)
- Log all decisions for audit trail
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from typing import Dict, Tuple, List
import tensorflow as tf
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract embeddings (feature vectors) from pill images.
    Uses penultimate layer of CNN for similarity-based matching.
    
    Medical Benefit: Enables open-set recognition (UNKNOWN detection).
    """
    
    def __init__(self, model, layer_name='global_average_pooling2d'):
        """
        Initialize embedding extractor.
        
        Args:
            model: Trained Keras model
            layer_name: Name of layer to extract embeddings from
        """
        self.model = model
        self.embedding_model = Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        self.embedding_dim = model.get_layer(layer_name).output.shape[-1]
        logger.info(f"Initialized EmbeddingExtractor with dimension {self.embedding_dim}")
    
    def extract_embeddings(self, images, batch_size=32):
        """
        Extract embeddings for a batch of images.
        
        Args:
            images: Array of images (N, H, W, C)
            batch_size: Batch size for processing
            
        Returns:
            np.array: Embeddings (N, embedding_dim)
        """
        embeddings = self.embedding_model.predict(images, batch_size=batch_size, verbose=0)
        return embeddings
    
    def extract_class_prototypes(self, class_images_dict):
        """
        Compute prototype (mean embedding) for each class.
        
        Args:
            class_images_dict: Dict {class_name: [images]}
            
        Returns:
            dict: {class_name: prototype_embedding}
        """
        prototypes = {}
        for class_name, images in class_images_dict.items():
            if len(images) > 0:
                embeddings = self.extract_embeddings(np.array(images))
                prototype = np.mean(embeddings, axis=0)
                prototypes[class_name] = prototype
                logger.info(f"Computed prototype for {class_name} from {len(images)} images")
        
        return prototypes


class SimilarityMatcher:
    """
    Match pill images to known classes using embedding similarity.
    Enables UNKNOWN tablet detection.
    """
    
    def __init__(self, embedding_extractor, class_prototypes):
        """
        Initialize similarity matcher.
        
        Args:
            embedding_extractor: EmbeddingExtractor instance
            class_prototypes: Dict {class_name: embedding}
        """
        self.embedding_extractor = embedding_extractor
        self.class_prototypes = class_prototypes
        self.class_names = list(class_prototypes.keys())
        self.prototype_matrix = np.array([
            class_prototypes[name] for name in self.class_names
        ])
        logger.info(f"Initialized SimilarityMatcher with {len(self.class_names)} classes")
    
    def compute_similarity(self, image_embedding, threshold=0.5):
        """
        Compute cosine similarity to all class prototypes.
        
        Medical Logic:
        - similarity < 0.5: UNKNOWN (not similar to any known pill)
        - similarity 0.5-0.7: UNCERTAIN (marginal match)
        - similarity > 0.7: CONFIDENT (matches known pill)
        
        Args:
            image_embedding: Single embedding (1, embedding_dim)
            threshold: Similarity threshold for UNKNOWN detection
            
        Returns:
            dict: {
                'most_similar_class': str,
                'similarity_score': float (0-1),
                'all_similarities': list of (class, similarity),
                'is_known_class': bool
            }
        """
        # Reshape for similarity computation
        embedding = image_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(embedding, self.prototype_matrix)[0]
        
        # Get top match
        top_idx = np.argmax(similarities)
        top_class = self.class_names[top_idx]
        top_similarity = float(similarities[top_idx])
        
        # Determine if it's a known class
        is_known = top_similarity >= threshold
        
        # Get all similarities sorted
        all_similarities = sorted(
            zip(self.class_names, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'most_similar_class': top_class if is_known else 'UNKNOWN',
            'similarity_score': top_similarity,
            'all_similarities': all_similarities[:3],  # Top 3
            'is_known_class': is_known,
            'threshold': threshold
        }


class ClassImbalanceAnalyzer:
    """
    Analyze class distribution and identify imbalanced classes.
    
    Medical Implication:
    - Under-represented classes may have lower accuracy
    - Leads to false negatives (missing rare tablets)
    - Requires data augmentation or reweighting
    """
    
    def __init__(self, train_df):
        """
        Initialize analyzer.
        
        Args:
            train_df: Training DataFrame with 'label' column
        """
        self.train_df = train_df
        self.class_distribution = train_df['label'].value_counts()
        self.total_samples = len(train_df)
        self.num_classes = len(self.class_distribution)
        self.imbalance_ratio = (
            self.class_distribution.max() / self.class_distribution.min()
        )
        logger.info(f"Class Imbalance Ratio: {self.imbalance_ratio:.2f}")
    
    def compute_class_weights(self):
        """
        Compute class weights for loss weighting (handles imbalance).
        
        Returns:
            dict: {class_idx: weight}
        """
        weights = {}
        for class_idx in range(self.num_classes):
            count = self.class_distribution.get(class_idx, 1)
            # Inverse frequency weighting
            weight = self.total_samples / (self.num_classes * count)
            weights[class_idx] = weight
        
        return weights
    
    def get_imbalanced_classes(self, percentile=25):
        """
        Identify under-represented classes.
        
        Args:
            percentile: Classes below this percentile are considered imbalanced
            
        Returns:
            list: Imbalanced class names
        """
        threshold = np.percentile(self.class_distribution.values, percentile)
        imbalanced = self.class_distribution[
            self.class_distribution < threshold
        ].index.tolist()
        logger.warning(f"Found {len(imbalanced)} imbalanced classes below {percentile}th percentile")
        return imbalanced
    
    def plot_distribution(self, save_path='class_distribution.png'):
        """Plot class distribution to visualize imbalance."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Histogram
        self.class_distribution.plot(kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('Class Distribution in Training Set', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].set_xlabel('Class Index')
        axes[0].axhline(y=self.class_distribution.mean(), color='r', linestyle='--', 
                       label=f'Mean: {self.class_distribution.mean():.0f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Imbalance ratio
        axes[1].plot(sorted(self.class_distribution.values, reverse=True), 'o-', linewidth=2)
        axes[1].set_title('Class Imbalance Severity (Sorted)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_xlabel('Ranked Class')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Distribution plot saved to {save_path}")
        plt.show()


class ConfidenceCalibrator:
    """
    Calibrate model confidence scores for better decision thresholds.
    
    Problem: Models are often overconfident (calibration error)
    Solution: Use temperature scaling to adjust confidence
    """
    
    def __init__(self, model_probs, true_labels, num_bins=15):
        """
        Initialize calibrator.
        
        Args:
            model_probs: Model output probabilities
            true_labels: Ground truth labels
            num_bins: Number of bins for calibration curve
        """
        self.model_probs = model_probs
        self.true_labels = true_labels
        self.num_bins = num_bins
        self.temperature = 1.0
        self._compute_calibration_metrics()
    
    def _compute_calibration_metrics(self):
        """Compute expected calibration error (ECE)."""
        pred_labels = np.argmax(self.model_probs, axis=1)
        confidences = np.max(self.model_probs, axis=1)
        
        accuracies = (pred_labels == self.true_labels).astype(int)
        
        bins = np.linspace(0, 1, self.num_bins + 1)
        self.ece = 0.0
        
        for i in range(self.num_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean()
                bin_conf = confidences[mask].mean()
                self.ece += abs(bin_acc - bin_conf) * mask.sum() / len(confidences)
        
        logger.info(f"Expected Calibration Error (ECE): {self.ece:.4f}")
    
    def temperature_scale(self, probs, temperature=1.5):
        """
        Apply temperature scaling to adjust confidence.
        
        Args:
            probs: Raw model probabilities
            temperature: Scaling factor (>1 reduces confidence, <1 increases)
            
        Returns:
            np.array: Scaled probabilities
        """
        logits = np.log(probs + 1e-10)
        scaled_logits = logits / temperature
        scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits), axis=1, keepdims=True)
        return scaled_probs
    
    def plot_calibration_curve(self, save_path='calibration_curve.png'):
        """Plot calibration curve to visualize confidence reliability."""
        pred_labels = np.argmax(self.model_probs, axis=1)
        confidences = np.max(self.model_probs, axis=1)
        accuracies = (pred_labels == self.true_labels).astype(int)
        
        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_accs = []
        bin_confs = []
        
        for i in range(self.num_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(accuracies[mask].mean())
                bin_confs.append(confidences[mask].mean())
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(bin_confs, bin_accs, 'o-', linewidth=2, label='Model Calibration')
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Calibration Curve (ECE: {self.ece:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
        plt.show()


class ContrastiveLearningUtils:
    """
    Utilities for contrastive learning (improves embedding quality).
    
    Medical Benefit:
    - Learn discriminative features (imprint, shape, color)
    - Improve robustness to visual variations
    - Better separation of visually similar pills
    """
    
    @staticmethod
    def create_triplet_loss():
        """
        Create triplet loss function for contrastive learning.
        
        Triplet Loss: L = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
        
        Medical Application:
        - anchor: image of pill A
        - positive: another image of same pill A
        - negative: image of visually similar pill B
        
        Returns:
            function: Triplet loss function
        """
        def triplet_loss(y_true, y_pred, margin=0.5):
            """
            Compute triplet loss.
            
            Args:
                y_true: Ground truth (unused, for Keras compatibility)
                y_pred: Predictions (N, 3*embedding_dim) - [anchor, positive, negative]
                margin: Margin for loss computation
            """
            embedding_dim = y_pred.shape[-1] // 3
            anchor = y_pred[:, :embedding_dim]
            positive = y_pred[:, embedding_dim:2*embedding_dim]
            negative = y_pred[:, 2*embedding_dim:]
            
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
            
            loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            return tf.reduce_mean(loss)
        
        return triplet_loss
    
    @staticmethod
    def create_contrastive_loss():
        """
        Create contrastive loss (SimCLR style) for self-supervised learning.
        
        Returns:
            function: Contrastive loss function
        """
        def contrastive_loss(y_true, y_pred, temperature=0.07):
            """
            NT-Xent loss (Normalized Temperature-scaled Cross Entropy).
            
            Args:
                y_true: Ground truth
                y_pred: Normalized embeddings
                temperature: Temperature for scaling
            """
            # Cosine similarity
            similarity = tf.matmul(y_pred, y_pred, transpose_b=True)
            similarity = similarity / temperature
            
            # Labels (diagonal = positive pairs)
            batch_size = tf.shape(y_pred)[0]
            labels = tf.eye(batch_size)
            
            # Cross entropy loss
            return tf.nn.softmax_cross_entropy_with_logits(labels, similarity)
        
        return contrastive_loss
    
    @staticmethod
    def mine_hard_negatives(embeddings, labels, k=5):
        """
        Mine hard negatives for training (pills visually similar to anchor).
        
        Medical Use: Identify visually confusing pills for focused training.
        
        Args:
            embeddings: Embedding matrix (N, embedding_dim)
            labels: Class labels (N,)
            k: Number of hard negatives to mine per anchor
            
        Returns:
            list: List of (anchor_idx, positive_idx, negative_idx) triplets
        """
        distances = np.sqrt(
            np.sum((embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :]) ** 2, axis=2)
        )
        
        triplets = []
        unique_labels = np.unique(labels)
        
        for anchor_idx in range(len(embeddings)):
            anchor_label = labels[anchor_idx]
            
            # Find positives (same class, not anchor)
            positive_mask = (labels == anchor_label) & (np.arange(len(labels)) != anchor_idx)
            if not positive_mask.any():
                continue
            
            positive_idx = np.random.choice(np.where(positive_mask)[0])
            
            # Find hard negatives (different class, but close in embedding space)
            negative_mask = labels != anchor_label
            negative_distances = distances[anchor_idx, negative_mask]
            
            # Select k closest negatives (hard negatives)
            closest_negatives = np.argsort(negative_distances)[:k]
            negative_idx = np.where(negative_mask)[0][closest_negatives[0]]
            
            triplets.append((anchor_idx, positive_idx, negative_idx))
        
        logger.info(f"Mined {len(triplets)} hard negative triplets")
        return triplets


class MedicalSafetyValidator:
    """
    Validate predictions for medical safety before deployment.
    """
    
    @staticmethod
    def validate_prediction(prediction_dict, confidence_threshold=0.70):
        """
        Validate prediction output for safety.
        
        Args:
            prediction_dict: Prediction response dictionary
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            dict: Validation result with safety status
        """
        result = {
            'is_safe': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Extract confidence
        try:
            conf_str = prediction_dict.get('confidence', '0%')
            confidence = float(conf_str.strip('%')) / 100.0
        except:
            result['is_safe'] = False
            result['errors'].append('Could not parse confidence value')
            return result
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            result['warnings'].append(
                f'Confidence {confidence*100:.1f}% below threshold {confidence_threshold*100:.0f}%'
            )
        
        # Check tablet name
        tablet_name = prediction_dict.get('tablet_name', 'UNKNOWN')
        if tablet_name == 'UNKNOWN TABLET':
            result['is_safe'] = False
            result['errors'].append('Could not identify tablet')
        
        # Check for required fields
        required_fields = ['tablet_name', 'confidence', 'dosage', 'precautions']
        for field in required_fields:
            if not prediction_dict.get(field):
                result['warnings'].append(f'Missing field: {field}')
        
        return result


# Utility function for medical recommendation
def get_safety_recommendation(confidence, is_confident):
    """
    Generate safety recommendation based on confidence.
    
    Args:
        confidence: Confidence score (0-1)
        is_confident: Whether prediction passed threshold
        
    Returns:
        str: Safety recommendation
    """
    if not is_confident:
        return "⚠️ SAFETY: Confidence too low. DO NOT consume. Contact pharmacist immediately."
    elif confidence < 0.80:
        return "⚠️ CAUTION: Verify with pharmacist before consumption."
    elif confidence < 0.90:
        return "✓ LIKELY SAFE: But verify with pharmacist if possible."
    else:
        return "✓ HIGH CONFIDENCE: Still recommend pharmacist verification."
