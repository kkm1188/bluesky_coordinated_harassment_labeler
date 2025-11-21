# Coordinated Harassment Labeler - Implementation Documentation

**Group Members:** Kendall Miller, Leo Li, Amy Chen, Tianyin Zhang

## Overview

This implementation detects **coordinated harassment campaigns** on Bluesky using a three-signal algorithmic approach that combines temporal, content similarity, and behavioral signals to identify potential pile-on attacks and organized harassment.

## Project Structure

```
bluesky_coordinated_harassment_labeler/
├── pylabel/                              # Main package
│   ├── __init__.py
│   ├── policy_proposal_labeler.py        # Coordination detection implementation
│   ├── automated_labeler.py              # Part 1: T&S words, news, dogs
│   └── label.py                          # Labeling utilities
├── test-data/                            # Test datasets
│   ├── data.csv                          # 150 labeled posts for evaluation
│   ├── input-posts-cite.csv              # Part 1 test data
│   ├── input-posts-dogs.csv
│   └── input-posts-t-and-s.csv
├── tests/                                # Test scripts
│   ├── test_coordination_detection.py    # Sanity check tests
│   ├── test_on_data.py                   # Test on data.csv
│   ├── performance_test.py               # Performance & scalability tests
│   └── evaluate_accuracy.py              # Accuracy evaluation
├── test-results/                         # Test output reports
│   ├── performance_report.txt            # Performance metrics
│   └── accuracy_evaluation.txt           # Accuracy, precision, recall
├── labeler-inputs/                       # Reference data
│   ├── t-and-s-words.csv
│   ├── t-and-s-domains.csv
│   └── news-domains.csv
└── README_COORDINATION_DETECTION.md      # This file
```

## Key Files

### `policy_proposal_labeler.py`
The core implementation of the coordinated harassment detection algorithm. Contains:
- `CoordinatedHarassmentLabeler` class
- Signal extraction functions (temporal, similarity, behavioral)
- Coordination score computation
- Label assignment logic
- Binary search optimization for temporal windows
- TF-IDF vectorization for content similarity

## Algorithm Design

### Label Taxonomy

The labeler applies one of three labels based on the coordination score:

| Label | Score Range | Description |
|-------|------------|-------------|
| `confirmed-coordination-high-risk` | 0.75 - 1.00 | High confidence coordination detected |
| `likely-coordination` | 0.60 - 0.75 | Probable coordination |
| `potential-coordination` | 0.40 - 0.60 | Possible coordination |
| (no label) | 0.00 - 0.40 | No coordination detected |

### Three-Signal Detection Algorithm

#### 1. Temporal Signal (Weight: 30%)

**What it detects:** Posts that mention the same target within a short time window.

**How it works:**
- Analyzes the time span of posts about the same target
- Posts concentrated within minutes receive higher scores
- Score decays as time span increases
- **Optimization:** Uses binary search to find posts within temporal windows (O(log N))
- Window: 10 minutes (configurable)

**Scoring:**
- < 1 minute: 1.0
- < 5 minutes: 0.8
- < 10 minutes: 0.4
- > 10 minutes: 0.2

**Example:**
- 4 posts within 2 minutes → High temporal score (0.8+)
- 4 posts over 30 minutes → Low temporal score (0.2)

#### 2. Content Similarity Signal (Weight: 50%)

**What it detects:** Repeated phrases or copy-paste style attacks.

**How it works:**
- **TF-IDF Vectorization:** Converts posts to numerical vectors using character n-grams (3-5 chars)
- **Cosine Similarity:** Computes similarity between all post pairs using vectorized operations
- Detects repeated phrases and common patterns
- Normalizes text (lowercase, remove URLs)
- Combined score: 40% TF-IDF similarity + 60% repeated phrase detection

**Technical Implementation:**
- Uses scikit-learn's TfidfVectorizer for efficiency
- Vectorized cosine similarity computation (NumPy)
- Handles vocabulary sizes up to ~10,000 terms

**Example:**
- Identical/near-identical posts → High similarity (0.9+)
- Similar themes, different wording → Medium similarity (0.4-0.7)
- Completely different content → Low similarity (0.0-0.3)

#### 3. Behavioral Signal (Weight: 20%)

**What it detects:** Suspicious account patterns suggesting coordinated activity.

**How it works:**
- Identifies new accounts (< 30 days old)
- Detects activity bursts (multiple posts from same account)
- Measures participation diversity (many different accounts)
- Analyzes account creation dates from CSV data

**Example:**
- 5 posts from 5 new accounts → High behavioral score (0.8+)
- 5 posts from 3 established accounts → Medium score (0.4)
- 5 posts from 5 old accounts → Low score (0.2)

### Coordination Score Computation

The final coordination score is a weighted average of the three signals:

```
score = (0.30 × temporal) + (0.50 × similarity) + (0.20 × behavioral)
```

**Rationale for weights:**
- **Similarity (50%):** Copy-paste attacks are the strongest indicator of coordination
- **Temporal (30%):** Clustering in time is strong evidence but can occur organically
- **Behavioral (20%):** Supporting evidence, but varies by scenario

### Algorithmic Optimizations

1. **Binary Search for Temporal Windows** - O(log N) instead of O(N) for finding posts in time window
2. **Timestamp Caching** - Parse timestamps once and cache to avoid redundant parsing
3. **Vectorized Similarity** - Use NumPy/sklearn for efficient matrix operations
4. **Two-Pass Batch Processing** - Build complete context before evaluation

## Installation & Setup

### Prerequisites

```bash
python >= 3.8
```

### Install Dependencies

```bash
cd bluesky_coordinated_harassment_labeler
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required Packages

```
atproto
pandas
numpy
scikit-learn
Pillow
requests
python-dotenv
```

## Usage

### Running Sanity Check Tests

To verify the implementation works correctly:

```bash
source venv/bin/activate
PYTHONPATH=. python3 tests/test_coordination_detection.py
```

**Expected output:**
- ✅ Coordinated Attack → `confirmed-coordination-high-risk` (score ~0.788)
- ✅ Mild Pile-On → `potential-coordination` (score ~0.433)
- ✅ Normal Criticism → No label (score ~0.174)
- ✅ Single Post → No label

### Testing on Labeled Dataset

To test on the 150-post dataset with ground truth:

```bash
source venv/bin/activate
PYTHONPATH=. python3 tests/test_on_data.py
```

### Running Performance Tests

To measure scalability and efficiency:

```bash
source venv/bin/activate
PYTHONPATH=. python3 tests/performance_test.py
```

This will test on datasets of 10, 100, 1,000, and 10,000 posts.

### Running Accuracy Evaluation

To calculate precision, recall, and F1 scores:

```bash
source venv/bin/activate
PYTHONPATH=. python3 tests/evaluate_accuracy.py
```

### Using the Labeler with Batch Data

```python
from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler
from datetime import datetime

# Initialize labeler
labeler = CoordinatedHarassmentLabeler(client=None)

# Prepare test data
posts = [
    {
        'post_id': '1',
        'author_id': 'user001',
        'timestamp': datetime.now().isoformat(),
        'text': '@target This is harassment!',
        'target_user': 'target',
        'author_created_at': datetime.now().isoformat()
    },
    # ... more posts
]

# Run detection
results = labeler.moderate_posts_batch(posts)

# Results: {'1': ['potential-coordination'], ...}
```

## Performance & Evaluation Results

### Complexity Analysis

**Overall Batch Processing: O(N log N)**
- N = total number of posts
- Dominated by timestamp sorting and binary search

**Per-Component Complexity:**
1. **Temporal Signal:** O(N log N + M) - Sorting + binary search per target
2. **Content Similarity:** O(M²) per target group where M = posts per target (typically M << N)
3. **Behavioral Signal:** O(M) - Linear scan through posts

**Memory Complexity:** O(N + V)
- N = number of posts stored
- V = vocabulary size for TF-IDF (typically < 10,000 terms)

### Performance Benchmarks

Tested on synthetic datasets with realistic human-like content:

| Dataset Size | Time per Post | Throughput | Total Time |
|-------------|---------------|------------|------------|
| 10 posts | 1.42 ms | 705 posts/s | 0.014s |
| 100 posts | 5.40 ms | 185 posts/s | 0.54s |
| 1,000 posts | 13.35 ms | 75 posts/s | 13.4s |
| 10,000 posts | 15.12 ms | 66 posts/s | 2.5 min |

**Memory Usage:** 3.34 MB peak for batch processing (150 posts)

**Key Findings:**
- ✅ Sub-quadratic scaling confirms O(N log N) complexity
- ✅ Suitable for real-time moderation on small batches
- ✅ Efficient batch processing for large datasets
- ✅ Very low memory footprint

### Accuracy Metrics

Evaluated on 150 labeled posts from `test-data/data.csv`:

**Binary Classification (Coordination vs No Coordination):**
- **Accuracy:** 72.00%
- **Precision:** 88.89% ⭐ (High - minimal false alarms)
- **Recall:** 51.95% (Conservative approach)
- **F1 Score:** 0.6557

**Confusion Matrix:**
- True Positives: 40 (correctly identified coordination)
- True Negatives: 68 (correctly identified no coordination)
- False Positives: 5 (only 5 false alarms!)
- False Negatives: 37 (missed coordination cases)

**Multi-class Accuracy:** 54.67% (exact label matching)

**Per-Class Performance:**

| Label | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| confirmed-coordination-high-risk | 100% | 20% | 0.33 |
| likely-coordination | 12.5% | 9.1% | 0.11 |
| potential-coordination | 26.1% | 24% | 0.25 |
| none | 64.8% | 93.2% | 0.76 |

**Trade-offs:**
- ✅ **High Precision (88.89%):** When it flags coordination, it's usually correct
- ✅ Minimizes false alarms → Better user experience
- ⚠️ **Lower Recall (51.95%):** Misses some coordination cases
- ⚠️ Conservative approach prioritizes precision over recall

## Implementation Details

### Key Functions

#### `moderate_posts_batch(posts_data: List[dict]) -> Dict[str, List[str]]`

Processes multiple posts from dataset (e.g., CSV) using optimized two-pass algorithm:

**Pass 1: Build Context**
- Parse and cache all timestamps (O(N))
- Sort posts by timestamp (O(N log N))
- Group posts by target user
- Create sorted timestamp-post pairs for binary search

**Pass 2: Evaluate Posts**
- For each post:
  - Use binary search to find posts in temporal window (O(log N))
  - Compute coordination score with full context
  - Assign labels based on thresholds

Returns mapping of post_id → labels

#### `compute_coordination_score(post: dict, group_context: List[dict], timestamp_cache: dict) -> float`

Computes the coordination score for a post:
- Calls signal extraction functions
- Combines signals with weights (30%, 50%, 20%)
- Returns score between 0 and 1

### Signal Extraction Functions

#### `_compute_temporal_signal(posts: List[dict], timestamp_cache: dict) -> float`
- Analyzes time distribution using cached timestamps
- Returns score based on temporal clustering
- Scoring: <1min=1.0, <5min=0.8, <10min=0.4, >10min=0.2

#### `_compute_content_similarity_signal(posts: List[dict]) -> float`
- Uses TF-IDF vectorization (scikit-learn)
- Computes cosine similarity matrix (vectorized)
- Detects repeated n-grams
- Returns combined similarity score (40% TF-IDF + 60% phrases)

#### `_compute_behavioral_signal(posts: List[dict]) -> float`
- Analyzes account creation dates
- Computes new account ratio (< 30 days)
- Detects activity bursts
- Measures participation diversity

### Configuration Parameters

Located in `policy_proposal_labeler.py`:

```python
# Label thresholds
THRESHOLD_POTENTIAL = 0.40
THRESHOLD_LIKELY = 0.60
THRESHOLD_CONFIRMED = 0.75

# Detection parameters
TEMPORAL_WINDOW_MINUTES = 10
NEW_ACCOUNT_DAYS = 30
SIMILARITY_THRESHOLD = 0.65
MIN_POSTS_FOR_COORDINATION = 2
```

## Testing & Evaluation

### Test Dataset (`test-data/data.csv`)

150 synthetic posts with ground truth labels, including:
- **30 posts** - confirmed-coordination-high-risk (identical/very similar harassment)
- **22 posts** - likely-coordination (similar themes, coordinated timing)
- **25 posts** - potential-coordination (weaker coordination signals)
- **73 posts** - none (organic posts, normal conversations)

**Edge Cases Included:**
- Bot-like behavior (identical WARNING messages)
- Activism campaigns (organized but legitimate)
- Single-user rapid posting (not coordinated)
- Sarcasm and criticism
- News event reactions
- Positive coordinated support
- Posts outside temporal window

### Automated Tests

All test scripts located in `tests/` directory:

1. **`test_coordination_detection.py`** - Sanity checks with hardcoded scenarios
2. **`test_on_data.py`** - Run labeler on full dataset
3. **`performance_test.py`** - Scalability and efficiency testing
4. **`evaluate_accuracy.py`** - Precision, recall, F1 metrics

Reports saved to `test-results/`:
- `performance_report.txt` - Detailed performance metrics
- `accuracy_evaluation.txt` - Accuracy analysis with error breakdown

## Known Limitations

### 1. Context Dependency
- Requires multiple posts (minimum 2) to detect coordination
- First post about a target cannot be labeled until more posts arrive
- Mitigated in batch mode with two-pass processing

### 2. Conservative Labeling
- High precision (88.89%) but lower recall (51.95%)
- Prioritizes avoiding false alarms over catching all cases
- May miss subtle or slow-burn coordination campaigns

### 3. False Negatives
- Misses ~48% of coordination cases in testing
- Struggles with:
  - First posts in a coordinated group (lack context)
  - Slower-paced campaigns (beyond 10-min window)
  - Posts with low content similarity but coordinated timing

### 4. False Positives
- Organic pile-ons may be flagged (e.g., many criticizing public figure)
- Legitimate activism with similar messaging
- News events causing rapid similar responses

### 5. Label Granularity
- Multi-class accuracy (54.67%) lower than binary (72%)
- Difficulty distinguishing between "likely" vs "potential" coordination
- Thresholds may need tuning per use case

## Future Improvements

### Algorithm Enhancements

1. **Adaptive Temporal Windows**
   - Detect campaign pacing automatically
   - Adjust window size based on target characteristics
   - Multi-scale temporal analysis

2. **Advanced NLP**
   - Use embeddings (BERT, sentence transformers) for semantic similarity
   - Detect paraphrased attacks
   - Multi-language support

3. **Network Analysis**
   - Analyze social graph patterns
   - Detect coordinated following/unfollowing
   - Identify bot networks using real account metadata

4. **Context-Aware Scoring**
   - Adjust thresholds based on target (public figure vs. private user)
   - Consider platform-wide activity levels
   - Learn from user feedback

### Performance Optimizations

5. **Locality-Sensitive Hashing**
   - Replace O(M²) similarity with approximate methods
   - Scale to larger target groups (M > 100)

6. **Streaming Algorithms**
   - Process posts incrementally
   - Maintain rolling temporal windows
   - Real-time detection

7. **Distributed Processing**
   - Parallel batch jobs for large datasets
   - Partition by target or time range

### Evaluation Improvements

8. **Larger Test Datasets**
   - Real Bluesky data with manual annotation
   - Cross-platform comparison (Twitter, Reddit)
   - Longitudinal studies of campaigns

9. **Threshold Optimization**
   - Grid search for optimal thresholds
   - ROC curve analysis
   - Precision-recall trade-off exploration

10. **Confidence Scores**
    - Output probability distributions
    - Support uncertain cases
    - Enable configurable sensitivity

## Ethical Considerations

### Privacy
- Uses publicly available metadata (post text, timestamps, usernames)
- No private data accessed
- Account creation dates from CSV (not ATProto API in test mode)
- In production: ensure compliance with privacy regulations

### False Positives
- Legitimate activism or organized advocacy may be flagged
- Labels are probabilistic ("potential", "likely") not definitive
- **High precision (88.89%)** minimizes but doesn't eliminate false alarms
- Human review recommended for enforcement decisions

### Transparency
- Algorithm is fully documented and explainable
- Individual signal scores available for debugging
- Users should understand why content is labeled
- Appeals process should be available

### Bias & Fairness
- Text-based heuristics may have linguistic bias
- Regular auditing needed to detect disparate impact
- Consider cultural and linguistic differences in communication patterns
- Test across diverse communities

### User Impact
- Conservative approach (high precision) balances safety with speech
- Tiered labels ("potential" → "confirmed") acknowledge uncertainty
- Users/moderators make final enforcement decisions
- Avoid over-reliance on automated systems

## Technical Approach Summary

This implementation demonstrates a **signal-based detection approach** that:

1. ✅ Combines multiple signals (temporal, content, behavioral)
2. ✅ Uses probabilistic scoring rather than binary classification
3. ✅ Provides explainable results (individual signal scores)
4. ✅ Acknowledges uncertainty in labeling (tiered labels)
5. ✅ Scales efficiently with O(N log N) complexity
6. ✅ Optimized with binary search and vectorized operations
7. ✅ Achieves high precision (88.89%) for reliability
8. ✅ Comprehensive testing with 150-post labeled dataset

The approach prioritizes **precision over recall** to minimize false positives, recognizing that incorrectly labeling legitimate speech is more harmful than missing some coordinated campaigns.

## Team Contributions

**Kendall Miller** - Core Detection Logic & Algorithm Translation
- Implemented three-signal detection algorithm
- Coordination score computation
- Label assignment logic

**Leo Li** - Algorithm Design Extensions & Performance Testing
- TF-IDF vectorization and cosine similarity
- Binary search optimization
- Performance testing suite (scalability, efficiency)
- Complexity analysis

**Amy Chen** - Dataset Creation & Evaluation Metrics
- Created 150-post labeled dataset
- Annotation rubric and ground truth labels
- Accuracy evaluation script (precision, recall, F1)

**Tianyin Zhang** - Policy Framing & Documentation
- Label taxonomy design
- Threshold definitions
- Ethical considerations
- README documentation

## Contact & Questions

For questions about:
- **Signal extraction functions** → See `policy_proposal_labeler.py` inline docs
- **Performance testing** → See `tests/performance_test.py`
- **Accuracy evaluation** → See `tests/evaluate_accuracy.py`
- **Dataset format** → See `test-data/data.csv`

---

**Last Updated:** 2025-11-20
**Version:** 2.0
**Assignment:** CS 5342 Assignment 3 - Coordinated Harassment Labeler
