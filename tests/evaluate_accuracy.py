#!/usr/bin/env python3
"""
Accuracy Evaluation Script for Coordinated Harassment Labeler
Evaluates precision, recall, F1, and accuracy against ground truth labels
Author: Leo Li - CS 5342 Assignment 3
"""

import csv
from collections import defaultdict
from datetime import datetime
from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler


def load_data_with_ground_truth(csv_file):
    """Load posts from CSV with ground truth labels"""
    posts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append(row)
    return posts


def normalize_label(label):
    """Normalize label strings for comparison"""
    if not label or label == 'none':
        return 'none'
    if 'confirmed' in label:
        return 'confirmed-coordination-high-risk'
    if 'likely' in label:
        return 'likely-coordination'
    if 'potential' in label:
        return 'potential-coordination'
    return 'none'


def calculate_metrics(ground_truth, predictions):
    """
    Calculate accuracy, precision, recall, and F1 score

    Returns metrics for:
    1. Binary classification (coordination vs no coordination)
    2. Multi-class classification (all label types)
    """

    # Binary metrics (any coordination vs none)
    binary_gt = ['coordination' if gt != 'none' else 'none' for gt in ground_truth]
    binary_pred = ['coordination' if pred != 'none' else 'none' for pred in predictions]

    tp_binary = sum(1 for gt, pred in zip(binary_gt, binary_pred) if gt == 'coordination' and pred == 'coordination')
    tn_binary = sum(1 for gt, pred in zip(binary_gt, binary_pred) if gt == 'none' and pred == 'none')
    fp_binary = sum(1 for gt, pred in zip(binary_gt, binary_pred) if gt == 'none' and pred == 'coordination')
    fn_binary = sum(1 for gt, pred in zip(binary_gt, binary_pred) if gt == 'coordination' and pred == 'none')

    binary_accuracy = (tp_binary + tn_binary) / len(ground_truth) if ground_truth else 0
    binary_precision = tp_binary / (tp_binary + fp_binary) if (tp_binary + fp_binary) > 0 else 0
    binary_recall = tp_binary / (tp_binary + fn_binary) if (tp_binary + fn_binary) > 0 else 0
    binary_f1 = 2 * (binary_precision * binary_recall) / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0

    # Multi-class metrics (exact label matching)
    exact_matches = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
    multiclass_accuracy = exact_matches / len(ground_truth) if ground_truth else 0

    # Per-class metrics
    labels = ['confirmed-coordination-high-risk', 'likely-coordination', 'potential-coordination', 'none']
    per_class_metrics = {}

    for label in labels:
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == label and pred == label)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt != label and pred == label)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == label and pred != label)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt != label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }

    return {
        'binary': {
            'accuracy': binary_accuracy,
            'precision': binary_precision,
            'recall': binary_recall,
            'f1': binary_f1,
            'tp': tp_binary,
            'tn': tn_binary,
            'fp': fp_binary,
            'fn': fn_binary
        },
        'multiclass': {
            'accuracy': multiclass_accuracy,
            'exact_matches': exact_matches,
            'total': len(ground_truth)
        },
        'per_class': per_class_metrics
    }


def analyze_errors(posts, ground_truth, predictions):
    """Analyze misclassifications to understand patterns"""

    false_positives = []
    false_negatives = []

    for i, (post, gt, pred) in enumerate(zip(posts, ground_truth, predictions)):
        gt_binary = 'coordination' if gt != 'none' else 'none'
        pred_binary = 'coordination' if pred != 'none' else 'none'

        if gt_binary == 'none' and pred_binary == 'coordination':
            false_positives.append({
                'post_id': post['post_id'],
                'text': post['text'],
                'ground_truth': gt,
                'predicted': pred,
                'target': post['target_user']
            })

        if gt_binary == 'coordination' and pred_binary == 'none':
            false_negatives.append({
                'post_id': post['post_id'],
                'text': post['text'],
                'ground_truth': gt,
                'predicted': pred,
                'target': post['target_user']
            })

    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def generate_report(metrics, errors, output_file):
    """Generate comprehensive evaluation report"""

    # Ensure test-results directory exists
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Always overwrite existing report
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COORDINATED HARASSMENT LABELER - ACCURACY EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Evaluator: Leo Li\n")
        f.write(f"Dataset: test-data/data.csv (150 posts with ground truth labels)\n")
        f.write("\n")

        # Binary Classification Metrics
        f.write("="*80 + "\n")
        f.write("BINARY CLASSIFICATION METRICS (Coordination vs No Coordination)\n")
        f.write("="*80 + "\n")
        binary = metrics['binary']
        f.write(f"Accuracy:  {binary['accuracy']:.4f} ({binary['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {binary['precision']:.4f} ({binary['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {binary['recall']:.4f} ({binary['recall']*100:.2f}%)\n")
        f.write(f"F1 Score:  {binary['f1']:.4f}\n")
        f.write("\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  True Positives:  {binary['tp']} (correctly identified coordination)\n")
        f.write(f"  True Negatives:  {binary['tn']} (correctly identified no coordination)\n")
        f.write(f"  False Positives: {binary['fp']} (flagged coordination incorrectly)\n")
        f.write(f"  False Negatives: {binary['fn']} (missed coordination)\n")
        f.write("\n")

        # Multi-class Metrics
        f.write("="*80 + "\n")
        f.write("MULTI-CLASS CLASSIFICATION METRICS (Exact Label Matching)\n")
        f.write("="*80 + "\n")
        multiclass = metrics['multiclass']
        f.write(f"Overall Accuracy: {multiclass['accuracy']:.4f} ({multiclass['accuracy']*100:.2f}%)\n")
        f.write(f"Exact Matches: {multiclass['exact_matches']}/{multiclass['total']}\n")
        f.write("\n")

        # Per-Class Metrics
        f.write("="*80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("="*80 + "\n")

        for label, m in metrics['per_class'].items():
            f.write(f"\n{label}:\n")
            f.write(f"  Precision: {m['precision']:.4f} ({m['precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {m['recall']:.4f} ({m['recall']*100:.2f}%)\n")
            f.write(f"  F1 Score:  {m['f1']:.4f}\n")
            f.write(f"  TP: {m['true_positives']}, FP: {m['false_positives']}, FN: {m['false_negatives']}, TN: {m['true_negatives']}\n")

        # Error Analysis
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("="*80 + "\n")

        f.write(f"\nFalse Positives: {len(errors['false_positives'])}\n")
        f.write("-" * 80 + "\n")
        for i, fp in enumerate(errors['false_positives'][:10], 1):  # Show first 10
            f.write(f"\n{i}. {fp['post_id']}\n")
            f.write(f"   Text: {fp['text'][:100]}...\n")
            f.write(f"   Ground Truth: {fp['ground_truth']}\n")
            f.write(f"   Predicted: {fp['predicted']}\n")
            f.write(f"   Target: {fp['target']}\n")

        if len(errors['false_positives']) > 10:
            f.write(f"\n... and {len(errors['false_positives']) - 10} more false positives\n")

        f.write(f"\n\nFalse Negatives: {len(errors['false_negatives'])}\n")
        f.write("-" * 80 + "\n")
        for i, fn in enumerate(errors['false_negatives'][:10], 1):  # Show first 10
            f.write(f"\n{i}. {fn['post_id']}\n")
            f.write(f"   Text: {fn['text'][:100]}...\n")
            f.write(f"   Ground Truth: {fn['ground_truth']}\n")
            f.write(f"   Predicted: {fn['predicted']}\n")
            f.write(f"   Target: {fn['target']}\n")

        if len(errors['false_negatives']) > 10:
            f.write(f"\n... and {len(errors['false_negatives']) - 10} more false negatives\n")

        f.write("\n" + "="*80 + "\n")


def main():
    """Run accuracy evaluation on data.csv"""

    print("="*80)
    print("COORDINATED HARASSMENT LABELER - ACCURACY EVALUATION")
    print("="*80)
    print()

    # Load data
    print("Loading test-data/data.csv...")
    posts = load_data_with_ground_truth('test-data/data.csv')
    print(f"Loaded {len(posts)} posts with ground truth labels\n")

    # Initialize labeler
    print("Initializing labeler...")
    labeler = CoordinatedHarassmentLabeler(client=None)

    # Run labeler
    print("Running labeler on all posts...")
    results = labeler.moderate_posts_batch(posts)

    # Extract ground truth and predictions
    ground_truth = []
    predictions = []

    for post in posts:
        gt = normalize_label(post['ground_truth_label'])
        ground_truth.append(gt)

        post_id = post['post_id']
        pred_labels = results.get(post_id, [])
        pred = normalize_label(pred_labels[0] if pred_labels else 'none')
        predictions.append(pred)

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(ground_truth, predictions)

    # Analyze errors
    print("Analyzing errors...")
    errors = analyze_errors(posts, ground_truth, predictions)

    # Display summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    binary = metrics['binary']
    print(f"\nBinary Classification (Coordination vs No Coordination):")
    print(f"  Accuracy:  {binary['accuracy']*100:.2f}%")
    print(f"  Precision: {binary['precision']*100:.2f}%")
    print(f"  Recall:    {binary['recall']*100:.2f}%")
    print(f"  F1 Score:  {binary['f1']:.4f}")

    multiclass = metrics['multiclass']
    print(f"\nMulti-class Accuracy (Exact Label Match):")
    print(f"  {multiclass['accuracy']*100:.2f}% ({multiclass['exact_matches']}/{multiclass['total']})")

    print(f"\nError Counts:")
    print(f"  False Positives: {len(errors['false_positives'])}")
    print(f"  False Negatives: {len(errors['false_negatives'])}")

    # Generate report
    output_file = 'test-results/accuracy_evaluation.txt'
    print(f"\nGenerating detailed report...")
    generate_report(metrics, errors, output_file)

    print(f"\n✓ Report saved to: {output_file}")
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
