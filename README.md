# Coordinated Harassment Labeler – Final Submission

**Course:** CS 5342 – Assignment 3  
**Repository:** `bluesky_coordinated_harassment_labeler`

**Group & IDs**  

- Kendall Miller — kkm88
- Leo Li — zk2262
- Amy Chen — ac3295
- Tianyin Zhang — tz445

---

## Submitted Files

### Core Labelers

| Path | Description |
| --- | --- |
| `pylabel/automated_labeler.py` | Part I labeler orchestrating the T&S keywords, news-domain, and perceptual dog detectors. |
| `pylabel/policy_proposal_labeler.py` | Coordinated harassment detector combining temporal, similarity, and behavioral signals (primary Part II deliverable). |
| `pylabel/label.py` | Shared helpers for fetching posts, input validation, and label-application utilities. |
| `pylabel/__init__.py` | Exposes `AutomatedLabeler` for compatibility with the harness. |

### Testing & Evaluation Scripts

| Path | Description |
| --- | --- |
| `tests/test_labeler.py` | Regression harness for Part I modes (dogs/news/T&S). |
| `tests/test_coordination_detection.py` | Scenario-based sanity checks for the coordination detector. |
| `tests/evaluate_accuracy.py` | Accuracy/precision/recall evaluator for `test-data/data.csv`; writes `test-results/accuracy_evaluation.txt`. |
| `tests/performance_test.py` | Stress test measuring latency, throughput, and memory; writes `test-results/performance_report.txt`. |

### Datasets & Sample Inputs

| Path | Description |
| --- | --- |
| `test-data/data.csv` | 150-post labeled dataset with ground-truth coordination labels. |
| `test-data/input-posts-dogs.csv` | Sample CSV for perceptual hash dog detector. |
| `test-data/input-posts-cite.csv` | Sample CSV for citation/news detection. |
| `test-data/input-posts-t-and-s.csv` | Sample CSV for T&S keyword detection. |

### Static Labeler Inputs

| Path | Description |
| --- | --- |
| `labeler-inputs/dog-list-images/` | 25 reference dog images (perceptual hash fingerprints). |
| `labeler-inputs/news-domains.csv` | Curated list of reputable news domains for citation detection. |
| `labeler-inputs/t-and-s-domains.csv` | Trust & safety domains blacklist. |
| `labeler-inputs/t-and-s-words.csv` | Canonical T&S keyword list used in Part I. |

### Generated Reports

| Path | Description |
| --- | --- |
| `test-results/accuracy_evaluation.txt` | Generated report with accuracy/precision/recall numbers (reproducible via script). |
| `test-results/performance_report.txt` | Generated report summarizing latency/throughput/memory metrics. |

### Documentation

| Path | Description |
| --- | --- |
| `README_COORDINATION_DETECTION.md` | In-depth design, ethics, weighting rationale, and known limitations. |
| `README_FINAL.md` | Submission wrapper with grading logistics. |
| `README_instructions.md` | Original Instructions for the Assignment |

### Utilities & Dependencies

| Path | Description |
| --- | --- |
| `get_post_test.py` | Utility for manual ingestion/testing during development. |
| `requirements.txt` | Locked Python dependencies (use with `pip install -r`). |

---

## Environment Setup (Step-by-Step)

1. **Install Python 3.8+** (tested on Python 3.9).  
2. **Clone and enter** the repo:  
   ```bash
   git clone https://github.com/kkm1188/bluesky_coordinated_harassment_labeler.git bluesky_coordinated_harassment_labeler
   cd bluesky_coordinated_harassment_labeler
   ```
3. **Create and activate** a virtual environment:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. (Optional) **Set PYTHONPATH** for ad‑hoc shells:  
   ```bash
   export PYTHONPATH=.
   ```

---

## Run the Labeler on Sample Posts

These commands assume the virtual environment is active and `PYTHONPATH=.`.

1. **Dog image detector** (perceptual hashing):  
   ```bash
   python tests/test_labeler.py labeler-inputs test-data/input-posts-dogs.csv
   ```
2. **News/citation detector**:  
   ```bash
   python tests/test_labeler.py labeler-inputs test-data/input-posts-cite.csv
   ```
3. **T&S word detector**:  
   ```bash
   python tests/test_labeler.py labeler-inputs test-data/input-posts-t-and-s.csv
   ```
4. **Coordinated harassment sanity scenarios**:  
   ```bash
   PYTHONPATH=. python tests/test_coordination_detection.py
   ```
   Expected qualitative outcomes:  
   - _Coordinated Attack_ → `confirmed-coordination-high-risk`  
   - _Mild Pile-On_ → `potential-coordination` (sometimes `likely`)  
   - _Normal Criticism_ / _Single Post_ → no label

---

## Evaluation Scripts (Reproduce Reports)

| Goal | Command | Output |
| --- | --- | --- |
| Accuracy metrics on 150-post dataset | `PYTHONPATH=. python tests/evaluate_accuracy.py` | Console summary + `test-results/accuracy_evaluation.txt` |
| Performance & scalability | `PYTHONPATH=. python tests/performance_test.py` | Console summary + `test-results/performance_report.txt` |

### Reproducing Report Numbers

After running `tests/evaluate_accuracy.py`, confirm the report shows (±0.01 tolerance due to randomness in timestamp parsing fallback):

- Binary accuracy ≈ **72.00%**  
- Precision ≈ **88.89%**, Recall ≈ **51.95%**, F1 ≈ **0.6557**  
- Multi-class accuracy ≈ **54.67%**  
- Confusion matrix: TP=40, TN=68, FP=5, FN=37

After running `tests/performance_test.py`, confirm highlights in `test-results/performance_report.txt`:

- Average batch latency ~**13.35 ms/post** for 1,000-post synthetic sets  
- Throughput around **75 posts/s** at that scale  
- Peak memory ≲ **3.5 MB** for the 150-post workload  
- Scalability table for datasets of 10, 100, 1,000, and 10,000 posts (as listed in the README_COORDINATION_DETECTION.md benchmarks)

---

## Notes on Code Documentation

- Core public methods in `pylabel/policy_proposal_labeler.py` include docstrings that align with the implemented temporal windowing, TF-IDF similarity, and behavioral heuristics.  
- Inline comments highlight the non-obvious pieces (timestamp caching, binary search boundaries, fallback similarity handling).  
- For deeper analysis or threshold rationales, refer to `README_COORDINATION_DETECTION.md`, which mirrors the production implementation—including weights, thresholds, and limitations.
