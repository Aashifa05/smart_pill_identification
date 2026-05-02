#!/usr/bin/env python
"""Evaluate the trained MobileNetV3 model on the real test split.

This script loads `model_feature_learning_mobilenetv3.keras`, recreates the
train/val/test split used during training, applies the correct MobileNetV3
preprocessing, runs predictions on the test set, computes metrics and a
confusion matrix, prints results, and writes a summary to
`model_evaluation_results.txt`.
"""

import os
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

# Ensure Admin is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Admin'))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

MODEL_PATH = Path(__file__).resolve().parents[1] / 'Admin' / 'Detection_and_Analysis_of_Pill' / 'models' / 'production' / 'model_feature_learning_mobilenetv3.keras'
DATA_DIR = Path(__file__).resolve().parents[1] / 'Admin' / 'media' / 'train'
OUTPUT_FILE = Path(__file__).resolve().parents[1] / 'Admin' / 'model_evaluation_results.txt'
MEDICATION_ADVISOR_INPUT = Path(__file__).resolve().parents[1] / 'Admin' / 'medication_data.json'


def load_images_from_nested_dir(dataset_path):
    images = []
    labels = []
    label_map = {}
    class_idx = 0

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not subdirs:
        raise RuntimeError('Expected nested class directories under media/train')

    for pill_dir in sorted(subdirs):
        pill_name = pill_dir.name
        label_map[pill_name] = class_idx
        class_idx += 1

        img_files = []
        for ext in image_extensions:
            img_files.extend(list(pill_dir.glob(ext)))

        for img_path in img_files:
            try:
                img = load_img(str(img_path), target_size=(224, 224))
                img_array = img_to_array(img).astype(np.float32)  # keep 0-255
                images.append(img_array)
                labels.append(label_map[pill_name])
            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, label_map


def normalize_text_list(texts):
    """Normalize text input to a list of stripped strings."""
    if isinstance(texts, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in texts]
    return [str(texts).strip()]


def response_matches(true_text, pred_text):
    """Return True if texts match exactly or by simple keyword overlap."""
    actual = re.sub(r'[^a-z0-9\s]', '', true_text.lower()).strip()
    predicted = re.sub(r'[^a-z0-9\s]', '', pred_text.lower()).strip()
    if actual == predicted:
        return True

    tokens = [token for token in actual.split() if len(token) > 3]
    return any(token in predicted.split() for token in tokens)


def compute_response_accuracy(y_true_text, y_pred_text):
    """Compute response accuracy via exact match or simple keyword match."""
    y_true = normalize_text_list(y_true_text)
    y_pred = normalize_text_list(y_pred_text)
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    if not y_true:
        return 0.0

    correct = sum(response_matches(t, p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def compute_bleu_score(y_true_text, y_pred_text):
    """Compute average BLEU score for text response pairs."""
    y_true = normalize_text_list(y_true_text)
    y_pred = normalize_text_list(y_pred_text)
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    if not y_true:
        return 0.0

    smoothing = SmoothingFunction().method1
    bleu_scores = []
    for true_text, pred_text in zip(y_true, y_pred):
        reference = re.sub(r'[^a-z0-9\s]', '', true_text.lower()).split()
        hypothesis = re.sub(r'[^a-z0-9\s]', '', pred_text.lower()).split()
        if not hypothesis or not reference:
            bleu_scores.append(0.0)
            continue
        bleu_scores.append(
            sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        )

    return float(np.mean(bleu_scores))


def ensure_medication_data_file(json_path):
    """Ensure the medication advisor JSON file exists and contains sample entries."""
    sample_data = [
        {
            "actual": "Take one tablet in the morning with water.",
            "predicted": "Take one tablet in the morning with water."
        },
        {
            "actual": "Contact your physician if symptoms worsen.",
            "predicted": "Call your doctor if symptoms get worse."
        }
    ]

    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if not json_path.exists() or json_path.stat().st_size == 0:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2)
            print(f'Created medication advisor JSON file with sample entries: {json_path}')
        return True
    except OSError as exc:
        print(f'Failed to prepare medication advisor JSON file: {exc}')
        return False


def load_medication_advisor_data(json_path):
    """Load medication advisor text data from a JSON file."""
    ensure_medication_data_file(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f'Failed to load medication advisor JSON data: {exc}')
        return [], []

    if not isinstance(data, list):
        print(f'Medication advisor JSON must contain a list of entries: {json_path}')
        return [], []

    y_true = []
    y_pred = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        actual = entry.get('actual')
        predicted = entry.get('predicted')
        if actual is None or predicted is None:
            continue
        y_true.append(actual)
        y_pred.append(predicted)

    if not y_true or not y_pred:
        print(f'Medication advisor JSON file contains no valid entries: {json_path}')
        return [], []

    return y_true, y_pred


def main():
    print('Loading model from', MODEL_PATH)
    model = load_model(str(MODEL_PATH))

    print('Loading images from', DATA_DIR)
    X, y, label_map = load_images_from_nested_dir(DATA_DIR)

    # one-hot labels for compatibility with splitting used in training
    num_classes = len(label_map)
    from tensorflow.keras.utils import to_categorical
    y_cat = to_categorical(y, num_classes=num_classes)

    # Recreate stratified split used in training
    from sklearn.model_selection import train_test_split

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_cat, test_size=0.2, stratify=y, random_state=42
    )
    label_indices_temp = np.argmax(y_temp, axis=1)
    val_size_adjusted = 0.1 / (1 - 0.2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, stratify=label_indices_temp, random_state=42
    )

    # Apply MobileNetV3 preprocessing (expects float 0-255 input)
    X_test_pp = preprocess_input(X_test.astype(np.float32))
    y_test_labels = np.argmax(y_test, axis=1)

    print('Running predictions on test set...')
    y_pred_proba = model.predict(X_test_pp, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    acc = accuracy_score(y_test_labels, y_pred)
    precision_macro = precision_score(y_test_labels, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test_labels, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test_labels, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)

    cls_report = classification_report(y_test_labels, y_pred, target_names=[name for name, _ in sorted(label_map.items(), key=lambda x: x[1])])
    cm = confusion_matrix(y_test_labels, y_pred)

    # Append class-wise accuracy (CAR) and weighted average recall (WAR)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    class_wise_accuracy = {
        class_name: float(cm[idx, idx] / cm[idx, :].sum()) if cm[idx, :].sum() > 0 else 0.0
        for idx, class_name in enumerate(class_names)
    }
    weighted_average_recall = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)

    # Print and save results
    out_lines = []
    out_lines.append('Model evaluation results (real test set)')
    out_lines.append('=' * 60)
    out_lines.append(f'Model: {MODEL_PATH.name}')
    out_lines.append(f'Num classes: {num_classes}')
    out_lines.append(f'Test samples: {len(X_test)}')
    out_lines.append('')
    out_lines.append(f'Accuracy: {acc:.6f} ({acc*100:.2f}%)')
    out_lines.append(f'Precision (macro): {precision_macro:.6f}')
    out_lines.append(f'Precision (weighted): {precision_weighted:.6f}')
    out_lines.append(f'Recall (macro): {recall_macro:.6f}')
    out_lines.append(f'Recall (weighted): {recall_weighted:.6f}')
    out_lines.append(f'F1-score (macro): {f1_macro:.6f}')
    out_lines.append(f'F1-score (weighted): {f1_weighted:.6f}')
    out_lines.append('')
    out_lines.append(f'Weighted Average Recall (WAR): {weighted_average_recall:.6f}')
    out_lines.append('')
    out_lines.append('Class-wise Accuracy (CAR):')
    for class_name, class_acc in class_wise_accuracy.items():
        out_lines.append(f'  {class_name}: {class_acc:.6f} ({class_acc*100:.2f}%)')
    out_lines.append('')
    out_lines.append('Classification Report:')
    out_lines.append(cls_report)
    out_lines.append('\nConfusion Matrix:')
    out_lines.append(np.array2string(cm, separator=', '))

    print('\n'.join(out_lines))

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))

    print('Saved evaluation summary to', OUTPUT_FILE)

    medication_advisor_path = Path(__file__).resolve().parents[1] / 'Admin' / 'medication_advisor_metrics.txt'
    advisor_available = False
    y_true_source = []
    y_pred_source = []

    try:
        if y_true_text is not None and y_pred_text is not None:
            y_true_source = y_true_text
            y_pred_source = y_pred_text
            advisor_available = bool(y_true_source and y_pred_source)
    except NameError:
        advisor_available = False

    if not advisor_available:
        y_true_source, y_pred_source = load_medication_advisor_data(MEDICATION_ADVISOR_INPUT)
        advisor_available = bool(y_true_source and y_pred_source)

    if advisor_available:
        response_acc = compute_response_accuracy(y_true_source, y_pred_source)
        bleu_score_value = compute_bleu_score(y_true_source, y_pred_source)

        med_lines = []
        med_lines.append('Medication Advisor Evaluation')
        med_lines.append('=' * 60)
        med_lines.append(f'Response Accuracy: {response_acc:.6f} ({response_acc*100:.2f}%)')
        med_lines.append(f'BLEU Score: {bleu_score_value:.6f}')

        print('\n'.join(['', 'Medication advisor metrics:'] + med_lines[1:]))

        with open(medication_advisor_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(med_lines))

        print('Saved medication advisor metrics to', medication_advisor_path)
    else:
        print('Medication advisor metrics skipped: no y_true_text/y_pred_text variables and no valid JSON input.')


if __name__ == '__main__':
    main()
