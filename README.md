# Coordinated Harassment Labeler – Final Submission

**Course:** CS 5342 – Assignment 3  
**Repository:** `bluesky_coordinated_harassment_labeler`

**Group & IDs**  
- Kendall Miller — Student ID:
- Leo Li — Student ID: 
- Amy Chen — Student ID: 
- Tianyin Zhang — Student ID: tz445

---

## Submitted Files (What to Grade)

| Path | Description |
| --- | --- |
| `pylabel/automated_labeler.py` | Part I labeler (T&S words, news domains, dog images). |
| `pylabel/policy_proposal_labeler.py` | Coordinated harassment detector with temporal, similarity, and behavioral signals. |
| `pylabel/label.py` | Shared helpers for fetching posts and applying labels. |
| `pylabel/__init__.py` | Package initializer. |
| `tests/test_labeler.py` | Provided harness for Part I regression tests. |
| `tests/test_coordination_detection.py` | Sanity checks for the coordination detector. |
| `tests/evaluate_accuracy.py` | Accuracy/precision/recall evaluator for `test-data/data.csv`. |
| `tests/performance_test.py` | Performance & scalability suite (latency, throughput, memory). |
| `test-data/data.csv` | 150-post labeled dataset with ground-truth coordination labels. |
| `test-data/input-posts-*.csv` | Part I sample CSV inputs (dogs, citations/news, T&S words). |
| `labeler-inputs/` | Static resources (dog reference images, news domains, T&S data). |
| `test-results/accuracy_evaluation.txt` | Generated accuracy report (should be reproduced by graders). |
| `test-results/performance_report.txt` | Generated performance report (should be reproduced by graders). |
| `README_COORDINATION_DETECTION.md` | Detailed design/ethics/analysis document. |
| `README_FINAL.md` | Submission README. |
| `requirements.txt` | Python dependency lock. |

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

- Every public method in `pylabel/policy_proposal_labeler.py` has a docstring describing real behavior (temporal binary-search windowing, TF-IDF similarity, behavioral heuristics).  
- Inline comments call out the non-obvious pieces (timestamp caching, binary search boundaries, fallback similarity handling).  
- If additional clarification is required, cross-reference `README_COORDINATION_DETECTION.md`, which mirrors the current implementation—including weights, thresholds, and known limitations.
