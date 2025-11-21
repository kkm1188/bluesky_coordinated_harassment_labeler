"""
Test script for coordinated harassment detection.
Provides sanity checks with example scenarios.
"""

from datetime import datetime, timedelta
from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler

def create_test_posts():
    """Create test post datasets for different scenarios."""

    base_time = datetime.now()

    # Scenario 1: Clear coordinated attack
    # - Multiple accounts
    # - Posts within minutes
    # - Very similar content (copy-paste)
    # - NEW accounts (created recently - suspicious!)
    coordinated_attack = [
        {
            'post_id': '1',
            'author_id': 'attacker001',
            'timestamp': base_time,
            'text': '@victim This person is a fraud and everyone should know it!',
            'target_user': 'victim',
            'author_created_at': base_time - timedelta(days=5)  # 5 day old account
        },
        {
            'post_id': '2',
            'author_id': 'attacker002',
            'timestamp': base_time + timedelta(seconds=30),
            'text': '@victim This person is a fraud and everyone should know it!',
            'target_user': 'victim',
            'author_created_at': base_time - timedelta(days=3)  # 3 day old account
        },
        {
            'post_id': '3',
            'author_id': 'attacker003',
            'timestamp': base_time + timedelta(minutes=1),
            'text': '@victim This person is a fraud and everyone should know it!',
            'target_user': 'victim',
            'author_created_at': base_time - timedelta(days=7)  # 7 day old account
        },
        {
            'post_id': '4',
            'author_id': 'user2024',
            'timestamp': base_time + timedelta(minutes=2),
            'text': '@victim Everyone should know this person is a fraud!',
            'target_user': 'victim',
            'author_created_at': base_time - timedelta(days=10)  # 10 day old account
        },
    ]

    # Scenario 2: Mild pile-on
    # - Multiple accounts
    # - Posts within reasonable time window
    # - Somewhat similar but not identical
    # - Mix of newer and established accounts
    mild_pile_on = [
        {
            'post_id': '10',
            'author_id': 'user_a',
            'timestamp': base_time,
            'text': '@celebrity I disagree with your statement about climate change',
            'target_user': 'celebrity',
            'author_created_at': base_time - timedelta(days=15)  # 15 day old account
        },
        {
            'post_id': '11',
            'author_id': 'user_b',
            'timestamp': base_time + timedelta(minutes=3),
            'text': '@celebrity Your take on climate change is problematic',
            'target_user': 'celebrity',
            'author_created_at': base_time - timedelta(days=20)  # 20 day old account
        },
        {
            'post_id': '12',
            'author_id': 'user_c',
            'timestamp': base_time + timedelta(minutes=5),
            'text': '@celebrity I think you should reconsider your climate views',
            'target_user': 'celebrity',
            'author_created_at': base_time - timedelta(days=45)  # 45 day old account (older)
        },
    ]

    # Scenario 3: Normal criticism (uncoordinated)
    # - Different accounts
    # - Spread out over time
    # - Different content
    # - Established accounts (not new/throwaway)
    normal_criticism = [
        {
            'post_id': '20',
            'author_id': 'random_user1',
            'timestamp': base_time,
            'text': '@politician I think your policy on education needs improvement',
            'target_user': 'politician',
            'author_created_at': base_time - timedelta(days=365)  # 1 year old account
        },
        {
            'post_id': '21',
            'author_id': 'random_user2',
            'timestamp': base_time + timedelta(minutes=15),
            'text': '@politician Just watched your speech, interesting points',
            'target_user': 'politician',
            'author_created_at': base_time - timedelta(days=500)  # ~1.5 year old account
        },
        {
            'post_id': '22',
            'author_id': 'random_user3',
            'timestamp': base_time + timedelta(minutes=30),
            'text': '@politician Have you considered alternative approaches to healthcare?',
            'target_user': 'politician',
            'author_created_at': base_time - timedelta(days=730)  # 2 year old account
        },
    ]

    # Scenario 4: Single post (no coordination possible)
    single_post = [
        {
            'post_id': '30',
            'author_id': 'lone_user',
            'timestamp': base_time,
            'text': '@someone I have some feedback about your project',
            'target_user': 'someone',
            'author_created_at': base_time - timedelta(days=100)  # 100 day old account
        },
    ]

    return {
        'coordinated_attack': coordinated_attack,
        'mild_pile_on': mild_pile_on,
        'normal_criticism': normal_criticism,
        'single_post': single_post
    }


def run_tests():
    """Run tests on different scenarios."""

    print("="*70)
    print("COORDINATED HARASSMENT DETECTION - SANITY CHECKS")
    print("="*70)
    print()

    # Initialize labeler (without actual client for testing)
    labeler = CoordinatedHarassmentLabeler(client=None)

    # Get test scenarios
    scenarios = create_test_posts()

    # Test each scenario
    for scenario_name, posts in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name.replace('_', ' ').upper()}")
        print(f"{'='*70}")

        print(f"\nNumber of posts: {len(posts)}")
        print("\nPost details:")
        for i, post in enumerate(posts, 1):
            print(f"  {i}. [{post['author_id']}] {post['text'][:60]}...")

        # Run detection
        results = labeler.moderate_posts_batch(posts)

        # Analyze results
        labels_found = set()
        for post_id, labels in results.items():
            if labels:
                labels_found.update(labels)

        print(f"\n{'─'*70}")
        print("RESULTS:")
        print(f"{'─'*70}")

        if not labels_found:
            print("  ✓ No coordination detected")
        else:
            print(f"  ⚠ Labels applied: {', '.join(labels_found)}")

        # Show detailed scores for first post (if applicable)
        if len(posts) >= 2:
            print(f"\n{'─'*70}")
            print("DETAILED ANALYSIS (last post):")
            print(f"{'─'*70}")

            last_post = posts[-1]
            target = last_post.get('target_user', '')

            if target:
                # Manually compute signals for demonstration
                all_posts = [p for p in posts if p.get('target_user') == target]

                temporal_score = labeler._compute_temporal_signal(all_posts)
                similarity_score = labeler._compute_content_similarity_signal(all_posts)
                behavioral_score = labeler._compute_behavioral_signal(all_posts)
                overall_score = labeler.compute_coordination_score(last_post, all_posts)

                print(f"  Temporal Signal:     {temporal_score:.3f}")
                print(f"  Similarity Signal:   {similarity_score:.3f}")
                print(f"  Behavioral Signal:   {behavioral_score:.3f}")
                print(f"  ─────────────────────────")
                print(f"  Overall Score:       {overall_score:.3f}")

                # Show threshold comparison
                print(f"\n  Thresholds:")
                print(f"    Potential:   {labeler._labels_from_score(0.40)[0] if overall_score >= 0.40 else '(not met)'} ≥ 0.40")
                print(f"    Likely:      {labeler._labels_from_score(0.60)[0] if overall_score >= 0.60 else '(not met)'} ≥ 0.60")
                print(f"    Confirmed:   {labeler._labels_from_score(0.75)[0] if overall_score >= 0.75 else '(not met)'} ≥ 0.75")

        print()

    # Expected results summary
    print("\n" + "="*70)
    print("EXPECTED RESULTS SUMMARY")
    print("="*70)
    print("""
    ✓ Coordinated Attack:      HIGH score → "confirmed-coordination-high-risk"
      - Very similar content (copy-paste)
      - Concentrated in time (< 2 minutes)
      - Multiple unique accounts

    ✓ Mild Pile-On:           MEDIUM score → "likely" or "potential"
      - Somewhat similar themes
      - Within temporal window but more spread out
      - Multiple accounts

    ✓ Normal Criticism:       LOW score → No label
      - Different content
      - Spread over longer time period
      - Organic discussion

    ✓ Single Post:            No label
      - Need minimum 3 posts for coordination analysis
    """)

    print("="*70)
    print("Testing complete!")
    print("="*70)


if __name__ == "__main__":
    run_tests()
