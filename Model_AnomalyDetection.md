# Anomaly Detection Model

## 1) Purpose 

In a Document Management System (DMS), the anomaly detection model’s purpose is to automatically spot documents, content, or user behaviors that deviate from normal patterns—such as corrupted or tampered scans, abnormal page/DPI or OCR-confidence profiles, duplicate uploads, out-of-distribution text, or unusual login/download spikes—so the system can take the right action (quarantine, route for human review, or allow) with an explainable risk score and reason tags.  Primary goals:
- **Input quality & integrity:** corrupted scans, abnormal DPI/page counts, duplicated uploads, low OCR confidence, tampered pages.
- **Content out‑of‑distribution (OOD):** text/images that deviate from the expected distribution for a document type (e.g., invoices, IDs, bank slips).
- **Fraud/tampering cues:** mismatched fonts, altered totals, pasted signatures/stamps (optional advanced phase).
- **System/usage anomalies:** sudden spikes in downloads/logins from an account/IP, unusual API call patterns (security analytics).


## 2) Use Cases (Input → Process → Output)

### 2.1 Document ingestion anomalies
**Input:** PDF, TIFF/PNG pages, metadata (size, pages, DPI, source).  
**Process:** extract structured features + run an unsupervised detector (IsolationForest / PyOD).  
**Output:** anomaly score ∈ [0,1], reason tags (e.g., `page_count_outlier`, `low_conf`), action (`quarantine`, `needs_review`).

### 2.2 Content (text) OOD
**Input:** OCR text / extracted text blocks by template.  
**Process:** embed with Sentence-Transformers → compare to known distribution (class centroids / kNN density / Mahalanobis).  
**Output:** distance score + OOD flag; optionally top‑k similar documents for reviewer context.

### 2.3 Image/forgery cues (optional Phase‑2)
**Input:** scanned images.  
**Process:** error‑level analysis heuristics or CNN‑based inconsistency detectors.  
**Output:** tamper score + regions of interest (heatmap).

### 2.4 Usage/security anomalies (streaming)
**Input:** event stream (logins, uploads, downloads, IPs).  
**Process:** online anomaly detection (Half‑Space Trees) + thresholds + rules.  
**Output:** alert events, risk scores, automatic rate limiting/block list suggestions.



## 3) Available Libraries (accuracy, speed, license)

| Library | Key Algos | Works Well For | Speed/Scale | License |
|---|---|---|---|---|
| **scikit‑learn** | IsolationForest, One‑Class SVM, LOF | classical tabular features | Fast on CPU; easy to tune | BSD‑3 |
| **PyOD** | ~50+ detectors incl. IForest, COPOD, ECOD, AutoEnc (DL) | broad unsupervised baseline suite | Good CPU; some DL via PyTorch/TensorFlow | MIT |
| **PyTorch / TensorFlow** | Autoencoders, VAE, CNN/LSTM | deep feature learning (images/text) | Needs GPU for speed | BSD‑style / Apache‑2.0 |
| **sentence‑transformers** | SBERT/mpnet embeddings | text OOD via distance | CPU OK; GPU faster | Apache‑2.0 |
| **alibi‑detect** | OOD, drift (KS, MMD, LLK, VAE) | robust OOD + drift tests | CPU/GPU | Apache‑2.0 |
| **river** | Half‑Space Trees (streaming), drift | online/streaming telemetry | Very fast incremental | MIT |

## 4) Pros & Cons

**Unsupervised (IForest/COPOD/ECOD)**  
- ✅ No labels needed; fast; simple thresholds.  
- ❌ Can flag rare but valid cases; needs per‑doctype normalization.

**Deep AEs / VAEs**  
- ✅ Capture complex patterns (images/text); configurable.  
- ❌ Needs more data/compute; careful training/validation to avoid overfitting.

**Text‑embedding OOD**  
- ✅ Strong for template‑style docs; interpretable via nearest neighbors.  
- ❌ Sensitive to domain shift and language; requires clean OCR/text extraction.

**Streaming HST**  
- ✅ Always‑on, low‑latency alerts for security/ops.  
- ❌ Requires feature engineering + drift handling.



## 5) Recommendation & Rationale (Phased)

**Phase‑1 (MVP, 2–3 weeks):**  
- L1 rules (hard checks): page_count, file_size, DPI ranges, OCR avg_conf, duplicate hash.  
- L2 model: **IsolationForest** (scikit‑learn) on normalized **ingestion features** per document type.  
- Text OOD pilot: centroid distance on **SBERT** embeddings per template.  
- Reviewer UI: show top contributing features and nearest neighbors.

**Phase‑2 (1–2 months):**  
- Replace/supplement IF with **PyOD COPOD/ECOD** ensemble; calibrate per‑doctype thresholds.  
- Add **alibi‑detect drift** monitors (detect distribution shift; auto‑retrain flag).  
- **Streaming security** with **river.HalfSpaceTrees** on login/download features.

**Phase‑3 (advanced):**  
- Autoencoder/VAE for image tamper cues (selected doc classes).  
- Human‑in‑the‑loop feedback store → weak labels → evaluate supervised anomaly classifiers.



## 6) Computational Efficiency

- **IForest:** O(n log n) train; O(trees · log n) predict. Comfortable on CPU for 10⁴–10⁵ samples, ~50–200 features.  
- **SBERT embeddings:** ~10–40 ms/page on CPU (fp32); GPU reduces 3–10× (batching helps).  
- **COPOD/ECOD:** linear time; very fast.  
- **HST (river):** constant‑time updates per event; ideal for streams.

Memory footprint is dominated by the embedding index (text OOD) and tree ensembles; use float16 where safe and persist models with joblib/onnx.



## 7) Implementation Notes 

-**Supported Libraries:** pandas, scikit-learn, joblib.

-**Input Data:** Parquet file (features_ingestion.parquet) containing numerical document features.

-**Preprocessing:** Standardization using StandardScaler.

-**Model:** IsolationForest (n_estimators=300, contamination=0.01, random_state=42).

-**Output Model:** Trained pipeline saved as models/iforest_ingestion.joblib.

-**Purpose:** Detect anomalies or outliers in document feature data.

**Example:**
### Training — IsolationForest (scikit‑learn)
```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_parquet("features_ingestion.parquet")  # one row per doc
X = df.drop(columns=["doc_id","dup_hash","created_at"]).select_dtypes("number")

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("iforest", IsolationForest(n_estimators=300, contamination=0.01, random_state=42))
])
pipe.fit(X)
joblib.dump(pipe, "models/iforest_ingestion.joblib")
```

## 8) Known Issues / Limitations
- Scarce labels; rely on reviewer feedback & synthetic anomalies.
- Domain shift across sources/scanners → drift monitoring required.
- OCR noise can inflate text OOD; always normalize text aggressively.
- Thresholds are doctype‑specific; use percentile tuning + guard rails.
- Privacy: store only hashed fingerprints where possible.

## 9) References 
- https://www.youtube.com/watch?v=OS9xRGKfx4E
- https://medium.com/@hassaanidrees7/anomaly-detection-techniques-and-applications-f7de41410883
- Donald, Jane & Banner, James & Satria, Robby & Tania, Winema & James, William. (2024). Anomaly detection in usage patterns.   
- https://www.wallarm.com/what/what-is-anomaly-detection



