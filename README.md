# Coordinated Harassment Labeler – Final Submission

**Course:** CS 5342 – Assignment 3  
**Repository:** `bluesky_coordinated_harassment_labeler`

**Group & IDs**  

- Kendall Miller — kkm88
- Leo Li — zl2262
- Amy Chen — ac3295
- Tianyin Zhang — tz445

---

## Submitted Files (Post-Starter Contributions)

### Coordination Detector & Tooling

| Path | Description |
| --- | --- |
| `pylabel/policy_proposal_labeler.py` | Final coordination detector (temporal, similarity, behavioral signals, caching, TF‑IDF scoring). |
| `tests/test_coordination_detection.py` | Scenario-based sanity suite covering attack, pile-on, and benign cases. |
| `tests/evaluate_accuracy.py` | Accuracy/precision/recall evaluator for the 150-post dataset; emits `test-results/accuracy_evaluation.txt`. |
| `tests/performance_test.py` | Performance harness measuring latency, throughput, memory; emits `test-results/performance_report.txt`. |
| `run.py` | Unified CLI to launch the coordination/accuracy/performance suites without touching the starter Part I harness. |

### Datasets, Reports, and Requirements

| Path | Description |
| --- | --- |
| `test-data/data.csv` | 150-post labeled dataset produced for coordination accuracy evaluation. |
| `test-results/accuracy_evaluation.txt` | Generated accuracy/precision/recall report. |
| `test-results/performance_report.txt` | Generated latency/throughput/memory report. |
| `requirements.txt` | Dependency lockfile for the coordination detector stack. |

### Documentation

| Path | Description |
| --- | --- |
| `README.md` | Submission write-up (this file). |
| `README_COORDINATION_DETECTION.md` | Detailed design doc with weights, heuristics, and limitations. |
| `README_FINAL.md` | Final submission checklist/readme. |
| `README_instructions.md` | Assignment instructions included for completeness. |

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
5. **Set PYTHONPATH** (required when running scripts outside `run.py`):  
   ```bash
   export PYTHONPATH=.
   ```
6. **Launch the unified CLI runner** to pick tests interactively (recommended):  

   ```bash
   python run.py
   ```

---

## Run the Coordination Detector

The recommended path is to run `python run.py` and select from the prompt:

| Menu | What it runs |
| --- | --- |
| `1` | `tests/test_coordination_detection.py` (scenario sanity suite) |
| `2` | `tests/evaluate_accuracy.py` |
| `3` | `tests/performance_test.py` |
| `a` | Runs all of the above sequentially |

If you prefer manual invocation (or need to capture stdout separately), activate the venv, set `PYTHONPATH=.`, and run:

```bash
PYTHONPATH=. python tests/test_coordination_detection.py
PYTHONPATH=. python tests/evaluate_accuracy.py
PYTHONPATH=. python tests/performance_test.py
```

Expected qualitative outcomes for `tests/test_coordination_detection.py`:  

- _Coordinated Attack_ → `confirmed-coordination-high-risk`  
- _Mild Pile-On_ → `potential-coordination`
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
