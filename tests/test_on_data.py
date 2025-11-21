#!/usr/bin/env python3
"""
Test the CoordinatedHarassmentLabeler on data.csv
"""

import csv
from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler

def load_data(csv_file):
    """Load posts from CSV"""
    posts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append(row)
    return posts

def main():
    print("="*70)
    print("Testing CoordinatedHarassmentLabeler on data.csv")
    print("="*70)

    # Load data
    print("\nLoading data.csv...")
    posts = load_data('data.csv')
    print(f"Loaded {len(posts)} posts")

    # Initialize labeler
    print("\nInitializing labeler...")
    labeler = CoordinatedHarassmentLabeler(client=None)

    # Run labeler
    print("\nRunning labeler on all posts...")
    results = labeler.moderate_posts_batch(posts)

    # Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Count predictions
    confirmed_count = 0
    likely_count = 0
    potential_count = 0
    none_count = 0

    for post_id, labels in results.items():
        if not labels:
            none_count += 1
        else:
            for label in labels:
                if 'confirmed' in label:
                    confirmed_count += 1
                elif 'likely' in label:
                    likely_count += 1
                elif 'potential' in label:
                    potential_count += 1

    print(f"\nPredicted Labels:")
    print(f"  confirmed-coordination-high-risk: {confirmed_count}")
    print(f"  likely-coordination: {likely_count}")
    print(f"  potential-coordination: {potential_count}")
    print(f"  none: {none_count}")

    # Count ground truth
    gt_confirmed = sum(1 for p in posts if p['ground_truth_label'] == 'confirmed-coordination-high-risk')
    gt_likely = sum(1 for p in posts if p['ground_truth_label'] == 'likely-coordination')
    gt_potential = sum(1 for p in posts if p['ground_truth_label'] == 'potential-coordination')
    gt_none = sum(1 for p in posts if p['ground_truth_label'] == 'none')

    print(f"\nGround Truth Labels:")
    print(f"  confirmed-coordination-high-risk: {gt_confirmed}")
    print(f"  likely-coordination: {gt_likely}")
    print(f"  potential-coordination: {gt_potential}")
    print(f"  none: {gt_none}")

    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)

    # Show first 10 posts
    for i, post in enumerate(posts[:10]):
        post_id = post['post_id']
        predicted = results.get(post_id, [])
        predicted_label = predicted[0] if predicted else 'none'
        ground_truth = post['ground_truth_label']

        match = "✓" if (predicted_label == ground_truth or
                        (not predicted and ground_truth == 'none')) else "✗"

        print(f"\n{match} {post_id}")
        print(f"  Text: {post['text'][:80]}...")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Predicted: {predicted_label}")

    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
