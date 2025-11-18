# Coordinated Harassment Labeler - Implementation Documentation

**Group Members:** Kendall Miller, Leo Li, Amy Chen, Tianyin Zhang

## Overview

This implementation detects **coordinated harassment campaigns** on Bluesky using a three-phase algorithmic approach that combines temporal, content similarity, and behavioral signals to identify potential pile-on attacks and organized harassment.

## Project Structure

```
bluesky-assign3/
├── pylabel/
│   ├── __init__.py
│   ├── policy_proposal_labeler.py   # Main coordination detection implementation
│   ├── automated_labeler.py         # Part 0 implementation (T&S words, news, dogs)
│   └── label.py                     # Labeling utilities
├── test_coordination_detection.py    # Sanity check tests
└── README_COORDINATION_DETECTION.md  # This file
```

## Key Files

### `policy_proposal_labeler.py`
The core implementation of the coordinated harassment detection algorithm. Contains:
- `CoordinatedHarassmentLabeler` class
- Signal extraction functions (temporal, similarity, behavioral)
- Coordination score computation
- Label assignment logic

## Algorithm Design

### Label Taxonomy

The labeler applies one of three labels based on the coordination score:

| Label | Score Range | Description |
|-------|------------|-------------|
| `confirmed-coordination-high-risk` | 0.75 - 1.00 | High confidence coordination detected |
| `likely-coordination` | 0.60 - 0.75 | Probable coordination |
| `potential-coordination` | 0.40 - 0.60 | Possible coordination |
| (no label) | 0.00 - 0.40 | No coordination detected |

### Three-Phase Detection Algorithm

#### 1. Temporal Signal (Weight: 30%)

**What it detects:** Posts that mention the same target within a short time window.

**How it works:**
- Analyzes the time span of posts about the same target
- Posts concentrated within minutes receive higher scores
- Score decays as time span increases
- Window: 10 minutes (configurable)

**Example:**
- 4 posts within 2 minutes → High temporal score (0.8+)
- 4 posts over 30 minutes → Low temporal score (0.2)

#### 2. Content Similarity Signal (Weight: 50%)

**What it detects:** Repeated phrases or copy-paste style attacks.

**How it works:**
- Computes pairwise text similarity between posts
- Detects common n-grams (3-grams and 4-grams)
- Normalizes text (lowercase, remove URLs, punctuation)
- Uses SequenceMatcher for similarity calculation

**Example:**
- Identical/near-identical posts → High similarity (0.9+)
- Similar themes, different wording → Medium similarity (0.4-0.7)
- Completely different content → Low similarity (0.0-0.3)

#### 3. Behavioral Signal (Weight: 20%)

**What it detects:** Suspicious account patterns suggesting coordinated activity.

**How it works:**
- Identifies new/throwaway accounts (heuristic-based)
- Detects activity bursts (multiple posts from same account)
- Measures participation diversity (many different accounts)

**New Account Heuristics:**
- Consecutive digits in username (e.g., user123, attacker001)
- Generic patterns (user*, account*, temp*, test*, anon*)
- Recent years in username (2020-2025)
- Underscores followed by numbers (user_456)
- Short usernames with numbers (<8 chars)

**Note:** In production, this would query actual account creation dates via ATProto API.

### Coordination Score Computation

The final coordination score is a weighted average of the three signals:

```
score = (0.30 × temporal) + (0.50 × similarity) + (0.20 × behavioral)
```

**Rationale for weights:**
- **Similarity (50%):** Copy-paste attacks are the strongest indicator of coordination
- **Temporal (30%):** Clustering in time is strong evidence but can occur organically
- **Behavioral (20%):** Supporting evidence, but heuristics are less reliable

## Installation & Setup

### Prerequisites

```bash
python >= 3.8
```

### Install Dependencies

```bash
cd bluesky-assign3
pip install -r requirements.txt
```

## Usage

### Running Sanity Check Tests

To verify the implementation works correctly:

```bash
python test_coordination_detection.py
```

**Expected output:**
- ✅ Coordinated Attack → `confirmed-coordination-high-risk`
- ✅ Mild Pile-On → `potential-coordination`
- ✅ Normal Criticism → No label
- ✅ Single Post → No label

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
        'timestamp': datetime.now(),
        'text': '@target This is harassment!',
        'target_user': 'target'
    },
    # ... more posts
]

# Run detection
results = labeler.moderate_posts_batch(posts)

# Results: {'1': ['potential-coordination'], ...}
```

### Using the Labeler with Individual Posts (URL-based)

```python
from pylabel.policy_proposal_labeler import CoordinatedHarassmentLabeler
from atproto import Client

# Initialize client
client = Client()
client.login(USERNAME, PASSWORD)

# Initialize labeler
labeler = CoordinatedHarassmentLabeler(client=client)

# Moderate a post
labels = labeler.moderate_post("https://bsky.app/profile/user/post/123")

print(f"Applied labels: {labels}")
```

## Implementation Details

### Key Functions

#### `moderate_post(url: str) -> List[str]`
- Processes a single post from URL
- Fetches post using ATProto client
- Identifies targets
- Gathers context (recent posts about same target)
- Computes coordination score
- Returns labels

#### `moderate_posts_batch(posts_data: List[dict]) -> Dict[str, List[str]]`
- Processes multiple posts from dataset (e.g., CSV)
- Uses two-pass processing:
  1. Build complete context for all targets
  2. Evaluate each post with full context
- Returns mapping of post_id → labels

#### `compute_coordination_score(post: dict, group_context: List[dict]) -> float`
- Computes the coordination score for a post
- Calls signal extraction functions
- Combines signals with weights
- Returns score between 0 and 1

### Signal Extraction Functions

#### `_compute_temporal_signal(posts: List[dict]) -> float`
- Analyzes time distribution of posts
- Returns score based on temporal clustering

#### `_compute_content_similarity_signal(posts: List[dict]) -> float`
- Computes pairwise text similarities
- Detects repeated phrases (n-grams)
- Returns combined similarity score

#### `_compute_behavioral_signal(posts: List[dict]) -> float`
- Analyzes account patterns
- Computes new account ratio
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

### Test Scenarios

The implementation has been tested against four scenarios:

1. **Coordinated Attack** (4 posts, <2 min, nearly identical text)
   - Expected: `confirmed-coordination-high-risk`
   - Result: ✅ Score 0.788

2. **Mild Pile-On** (3 posts, 5 min, similar themes)
   - Expected: `potential-coordination`
   - Result: ✅ Score 0.433

3. **Normal Criticism** (3 posts, 30 min, different content)
   - Expected: No label
   - Result: ✅ Score 0.174

4. **Single Post**
   - Expected: No label (requires 2+ posts)
   - Result: ✅ Correctly skipped

### Performance Characteristics

- **Minimum posts required:** 2 (coordination requires multiple actors)
- **Temporal window:** 10 minutes
- **Complexity:** O(N²) for similarity computation (N = posts in window)
- **Memory:** Maintains cache of recent posts per target

## Known Limitations

### 1. Account Age Detection
- Uses heuristics (username patterns) instead of real account creation dates
- In production, should query ATProto API for actual account metadata
- May have false positives/negatives

### 2. Context Dependency
- Requires multiple posts to detect coordination
- First post about a target cannot be labeled until more posts arrive
- Fixed in batch mode with two-pass processing

### 3. Similarity Computation
- O(N²) complexity for pairwise comparisons
- For very large batches (100+ posts), may need optimization
- Could use min-hashing or LSH for scalability

### 4. False Positives
- Organic pile-ons (e.g., many people criticizing a public figure) may be flagged
- Labels are intentionally phrased as "potential" / "likely" to avoid overclaiming
- Users/moderators make final enforcement decisions

### 5. Temporal Assumptions
- Fixed 10-minute window may not capture all coordination patterns
- Some campaigns may be slower/faster
- Future work: adaptive window sizes

## Future Improvements

1. **Real Account Metadata**
   - Query ATProto for actual account creation dates
   - Analyze follower graphs for coordinated networks
   - Track account activity history

2. **Advanced NLP**
   - Use embeddings (BERT, sentence transformers) for semantic similarity
   - Detect paraphrased attacks
   - Multi-language support

3. **Network Analysis**
   - Analyze social graph patterns
   - Detect coordinated following/unfollowing
   - Identify bot networks

4. **Adaptive Thresholds**
   - Context-aware scoring (public figures vs. private users)
   - Adjust thresholds based on target characteristics
   - Learn from user feedback

5. **Performance Optimization**
   - Implement locality-sensitive hashing for similarity
   - Use streaming algorithms for large-scale data
   - Parallel processing for batch jobs

## Ethical Considerations

### Privacy
- The labeler uses publicly available metadata (post text, timestamps, usernames)
- No private data is accessed
- In production, ensure compliance with privacy regulations

### False Positives
- Legitimate activism or organized advocacy may be flagged
- Labels are probabilistic ("potential", "likely") not definitive
- Human review recommended for enforcement decisions

### Transparency
- Algorithm is fully documented and explainable
- Users should understand why content is labeled
- Appeals process should be available

### Bias & Fairness
- Heuristics may disproportionately flag certain communities
- Regular auditing needed to detect bias
- Consider cultural and linguistic differences

## Technical Approach Summary

This implementation demonstrates a **signal-based detection approach** that:

1. ✅ Combines multiple signals (temporal, content, behavioral)
2. ✅ Uses probabilistic scoring rather than binary classification
3. ✅ Provides explainable results (individual signal scores)
4. ✅ Acknowledges uncertainty in labeling (tiered labels)
5. ✅ Scales to batch processing (two-pass algorithm)

The approach prioritizes **precision** over **recall** to minimize false positives, recognizing that incorrectly labeling legitimate speech is more harmful than missing some coordinated campaigns.

## Contact & Questions

**Implementation Lead (Kendall):** Core detection logic & algorithm translation

For questions about:
- Signal extraction functions
- Coordination score computation
- Algorithm design choices

Please refer to inline documentation in `policy_proposal_labeler.py`.

---

**Last Updated:** 2025-11-15
**Version:** 1.0
**Assignment:** CS 5342 Assignment 3 - Coordinated Harassment Labeler
