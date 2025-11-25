#!/usr/bin/env python3

import time
import tracemalloc
import csv
import statistics
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import random

from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler


class PerformanceTester:
    """Performance testing harness for CoordinatedHarassmentLabeler"""

    def __init__(self):
        # Initialize labeler without actual Bluesky client for performance testing
        self.labeler = CoordinatedHarassmentLabeler(client=None)
        self.results = {}

    def load_test_data(self, csv_file: str) -> List[Dict]:
        """Load posts from CSV file"""
        posts = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                posts.append(row)
        return posts

    def generate_stress_test_data(self, num_posts: int, scenario: str = "coordinated") -> List[Dict]:
        """
        Generate synthetic test data for stress testing with realistic human-like content

        Args:
            num_posts: Number of posts to generate
            scenario: Type of scenario ("coordinated", "normal", "mixed")
        """
        posts = []
        base_time = datetime.now()

        # Realistic coordinated attack templates with variations
        coordinated_templates = [
            [
                "@{target} is spreading dangerous misinformation and people need to know about this. I've seen the evidence myself and it's concerning.",
                "@{target} spreading dangerous misinformation that people should know about. I've personally seen the evidence and it's really concerning.",
                "@{target} is spreading misinformation that's dangerous and everyone needs to be aware. I've seen the evidence and it's very concerning.",
                "@{target} dangerous misinformation being spread here. People need to know this. I've seen all the evidence and it's extremely concerning.",
            ],
            [
                "@{target} your behavior is completely unacceptable and crosses the line. The community deserves better than this kind of conduct.",
                "@{target} this behavior is totally unacceptable and way over the line. Our community deserves so much better than this.",
                "@{target} completely unacceptable behavior that crosses every line. The community really deserves better than what you're doing.",
                "@{target} this is unacceptable behavior that's crossed the line. The community deserves better treatment than this honestly.",
            ],
            [
                "@{target} please stop with the constant lies and misinformation already. Everyone can see through it at this point.",
                "@{target} stop lying and spreading misinformation constantly. Everyone sees right through it now.",
                "@{target} the constant lies and misinformation need to stop. Everyone can see through this nonsense.",
                "@{target} stop the lies and misinformation right now. Everyone's seeing through it at this point.",
            ],
        ]

        # Realistic normal post templates
        normal_templates = [
            "@{target} really enjoyed your latest {topic} post! The {detail} was particularly insightful. Looking forward to more content like this.",
            "@{target} I have to respectfully disagree with your take on {topic}. Have you considered {detail}? Would love to hear your thoughts.",
            "@{target} great work on the {topic} project! The {detail} shows real attention to detail. Keep it up!",
            "@{target} quick question about your {topic} post from yesterday. Could you clarify what you meant by {detail}?",
            "@{target} just finished reading your article on {topic}. The section about {detail} was fascinating. Thanks for sharing!",
            "@{target} your perspective on {topic} is interesting but I think there's more to consider. What about {detail}?",
            "@{target} loved the {topic} thread you posted. The part about {detail} really resonated with me.",
            "@{target} been following your work on {topic} for a while now. The {detail} approach is really innovative.",
        ]

        # Varied topics and details for organic posts
        topics = ["climate policy", "healthcare reform", "AI ethics", "urban planning", "education funding",
                 "renewable energy", "data privacy", "tech regulation", "housing policy", "transportation"]
        details = ["methodology", "data analysis", "implementation strategy", "case studies", "research findings",
                  "cost-benefit analysis", "stakeholder input", "long-term impacts", "alternative approaches", "timeline"]

        if scenario == "coordinated":
            # Heavy coordination: multiple users, same target, close timing, similar content
            target = "victim_user"
            template_group = random.choice(coordinated_templates)

            for i in range(num_posts):
                # Add some natural variation to timing
                time_offset = i * random.randint(3, 8)  # 3-8 seconds apart
                text = template_group[i % len(template_group)].format(target=f"@{target}")

                posts.append({
                    'post_id': f'stress_post_{i}',
                    'author_id': f'author_{i % 25}',  # 25 different authors for variety
                    'timestamp': (base_time + timedelta(seconds=time_offset)).isoformat(),
                    'text': text,
                    'target_user': target,
                    'author_created_at': (base_time - timedelta(days=random.randint(2, 15))).isoformat(),  # New accounts
                })

        elif scenario == "normal":
            # Normal traffic: varied targets, spread out in time, diverse unique content
            targets = ["person1", "person2", "person3", "person4", "person5", "person6", "person7"]

            for i in range(num_posts):
                target = random.choice(targets)
                template = random.choice(normal_templates)
                topic = random.choice(topics)
                detail = random.choice(details)

                text = template.format(target=f"@{target}", topic=topic, detail=detail)

                posts.append({
                    'post_id': f'normal_post_{i}',
                    'author_id': f'user_{i}',
                    'timestamp': (base_time + timedelta(minutes=i * random.randint(5, 15))).isoformat(),  # 5-15 min apart
                    'text': text,
                    'target_user': target,
                    'author_created_at': (base_time - timedelta(days=random.randint(90, 800))).isoformat(),  # Established accounts
                })

        elif scenario == "mixed":
            # Mix of coordinated and normal - more realistic distribution
            num_coordinated = num_posts // 3
            num_normal = num_posts - num_coordinated

            coord_posts = self.generate_stress_test_data(num_coordinated, "coordinated")
            normal_posts = self.generate_stress_test_data(num_normal, "normal")
            posts = coord_posts + normal_posts

            # Re-sort by timestamp after shuffling to maintain temporal order
            random.shuffle(posts)
            posts.sort(key=lambda p: p['timestamp'])

        return posts

    def test_single_post_latency(self, posts: List[Dict], num_samples: int = 50) -> Dict:
        """
        Measure time to process individual posts via batch processing

        Note: The labeler is designed for batch mode, so we test small batches
        """
        print(f"\n{'='*60}")
        print("TEST 1: Single Post Latency (via small batches)")
        print(f"{'='*60}")

        latencies = []
        sample_posts = random.sample(posts, min(num_samples, len(posts)))

        for post in sample_posts:
            start = time.perf_counter()
            # Process as a single-item batch
            _ = self.labeler.moderate_posts_batch([post])
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds

        results = {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'samples': len(latencies)
        }

        print(f"Samples tested: {results['samples']}")
        print(f"Mean latency: {results['mean']:.3f} ms")
        print(f"Median latency: {results['median']:.3f} ms")
        print(f"Min latency: {results['min']:.3f} ms")
        print(f"Max latency: {results['max']:.3f} ms")
        print(f"Std deviation: {results['stdev']:.3f} ms")

        self.results['single_post'] = results
        return results

    def test_batch_processing(self, posts: List[Dict]) -> Dict:
        """
        Measure time to process a full batch of posts
        """
        print(f"\n{'='*60}")
        print("TEST 2: Batch Processing Performance")
        print(f"{'='*60}")

        num_posts = len(posts)
        print(f"Processing {num_posts} posts in batch mode...")

        # Start memory tracking
        tracemalloc.start()

        start = time.perf_counter()
        labels_dict = self.labeler.moderate_posts_batch(posts)
        end = time.perf_counter()

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_time = end - start
        avg_time_per_post = (total_time / num_posts) * 1000  # ms per post
        throughput = num_posts / total_time  # posts per second

        results = {
            'total_posts': num_posts,
            'total_time_sec': total_time,
            'avg_time_per_post_ms': avg_time_per_post,
            'throughput_posts_per_sec': throughput,
            'memory_current_mb': current / (1024 * 1024),
            'memory_peak_mb': peak / (1024 * 1024),
            'labels_assigned': sum(1 for labels in labels_dict.values() if labels)
        }

        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average time per post: {avg_time_per_post:.3f} ms")
        print(f"Throughput: {throughput:.2f} posts/second")
        print(f"Memory used: {results['memory_current_mb']:.2f} MB")
        print(f"Peak memory: {results['memory_peak_mb']:.2f} MB")
        print(f"Posts labeled: {results['labels_assigned']}/{num_posts}")

        self.results['batch_processing'] = results
        return results

    def test_scalability(self, scenarios: Dict[str, int]) -> Dict:
        """
        Test how performance scales with different dataset sizes

        Args:
            scenarios: Dict mapping scenario names to post counts
                      e.g., {"small": 50, "medium": 150, "large": 500}
        """
        print(f"\n{'='*60}")
        print("TEST 3: Scalability Analysis")
        print(f"{'='*60}")

        results = {}

        for scenario_name, num_posts in scenarios.items():
            print(f"\n--- Scenario: {scenario_name} ({num_posts} posts) ---")

            # Generate coordinated attack scenario for stress testing
            posts = self.generate_stress_test_data(num_posts, "coordinated")

            start = time.perf_counter()
            labels_dict = self.labeler.moderate_posts_batch(posts)
            end = time.perf_counter()

            total_time = end - start
            avg_time = (total_time / num_posts) * 1000

            results[scenario_name] = {
                'num_posts': num_posts,
                'total_time_sec': total_time,
                'avg_time_per_post_ms': avg_time,
                'throughput': num_posts / total_time
            }

            print(f"  Time: {total_time:.3f}s | Avg: {avg_time:.3f}ms/post | Throughput: {num_posts/total_time:.2f} posts/s")

        self.results['scalability'] = results
        return results

    def test_worst_case_scenario(self) -> Dict:
        """
        Test worst-case performance: highly coordinated attack with many posts
        """
        print(f"\n{'='*60}")
        print("TEST 4: Worst-Case Scenario (Heavy Coordination)")
        print(f"{'='*60}")

        # Generate a scenario with maximum coordination signals:
        # - All posts within 1 minute
        # - Same target
        # - Very similar content
        # - New accounts
        num_posts = 100
        posts = []
        base_time = datetime.now()
        target = "@victim"
        same_content = f"{target} is terrible and should be banned immediately"

        for i in range(num_posts):
            posts.append({
                'post_id': f'worst_case_{i}',
                'author_id': f'author_{i}',
                'timestamp': (base_time + timedelta(seconds=i * 2)).isoformat(),  # 2 sec apart
                'text': same_content,  # Identical text
                'target_user': target,
                'author_created_at': (base_time - timedelta(days=5)).isoformat(),  # Very new accounts
            })

        start = time.perf_counter()
        labels_dict = self.labeler.moderate_posts_batch(posts)
        end = time.perf_counter()

        total_time = end - start

        # Check detection accuracy
        labeled_posts = [pid for pid, labels in labels_dict.items() if labels]

        results = {
            'num_posts': num_posts,
            'total_time_sec': total_time,
            'avg_time_per_post_ms': (total_time / num_posts) * 1000,
            'detection_rate': len(labeled_posts) / num_posts,
            'labels_assigned': len(labeled_posts)
        }

        print(f"Processing time: {total_time:.3f}s")
        print(f"Avg time per post: {results['avg_time_per_post_ms']:.3f}ms")
        print(f"Detection rate: {results['detection_rate']*100:.1f}% ({len(labeled_posts)}/{num_posts} posts labeled)")

        # Sample some labels
        if labeled_posts:
            sample = random.choice(labeled_posts)
            print(f"Sample label: {labels_dict[sample]}")

        self.results['worst_case'] = results
        return results

        print(analysis)
        self.results['complexity'] = analysis

    def generate_report(self, output_file: str = "performance_report.txt") -> None:
        """Generate a comprehensive performance report"""
        # Ensure test-results directory exists
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Always overwrite existing report
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COORDINATED HARASSMENT LABELER - PERFORMANCE TEST REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("\n")

            # Write all results
            for test_name, test_results in self.results.items():
                f.write(f"\n{test_name.upper().replace('_', ' ')}\n")
                f.write("-" * 70 + "\n")
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                else:
                    f.write(str(test_results) + "\n")

            f.write("\n" + "="*70 + "\n")

        print(f"\n\nReport saved to: {output_file}")


def main():
    """Run comprehensive performance tests"""
    print("\n" + "="*70)
    print("COORDINATED HARASSMENT LABELER - PERFORMANCE TESTING SUITE")
    print("="*70)

    tester = PerformanceTester()

    # Test 1: Generate test data and measure single-post latency
    print("\nGenerating test data...")
    test_posts = tester.generate_stress_test_data(150, "mixed")
    tester.test_single_post_latency(test_posts, num_samples=50)

    # Test 2: Batch processing performance
    tester.test_batch_processing(test_posts)

    # Test 3: Scalability across different dataset sizes
    tester.test_scalability({
        "tiny": 10,
        "small": 100,
        "medium": 1000,
        "large": 10000
    })

    # Test 4: Worst-case scenario
    tester.test_worst_case_scenario()

    # Generate final report
    tester.generate_report("test-results/performance_report.txt")

    print("\n" + "="*70)
    print("PERFORMANCE TESTING COMPLETE")
    print("="*70)
    print("\nKey Findings Summary:")
    if 'batch_processing' in tester.results:
        bp = tester.results['batch_processing']
        print(f"  • Batch Processing: {bp['avg_time_per_post_ms']:.2f} ms/post")
        print(f"  • Throughput: {bp['throughput_posts_per_sec']:.2f} posts/second")
        print(f"  • Memory Usage: {bp['memory_peak_mb']:.2f} MB peak")

    print("\n✓ All tests completed successfully!")
    print("✓ Report saved to test-results/performance_report.txt")


if __name__ == "__main__":
    main()
