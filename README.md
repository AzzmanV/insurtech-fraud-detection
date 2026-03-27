# Detecting the Invisible: AI-Powered Fraud Detection for Device Protection Insurance

> **AIM Postgraduate Diploma in Artificial Intelligence & Machine Learning — Pillar 5 Capstone Project**

**Author:** Mark Eliezer M. Villola | Country General Manager, ProTech Devices Philippines | [protechdevices.com](https://www.protechdevices.com)  
**Submitted:** March 2026 | **Programme:** Asian Institute of Management

---

## The Problem

Device protection insurance fraud in Asia follows three dominant patterns — claims filed within days of policy activation, retail partners with anomalous submission rates, and payout amounts that approach or exceed device replacement value. At ProTech Devices Asia, approximately 400–600 claims are processed monthly with no systematic fraud scoring at intake. Suspicious claims are flagged by adjuster intuition alone.

This project builds an end-to-end machine learning pipeline that assigns a fraud probability score to every claim at submission — before any adjuster review begins — enabling intelligent routing, automated approvals for clearly legitimate claims, and concentrated human review on the highest-risk cases.

---

## Results at a Glance

| Model | AUC-ROC | PR-AUC | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|---|---|---|---|---|
| **XGBoost ★ Champion** | **0.9412** | **0.7318** | **0.76** | **0.79** | **0.77** |
| Random Forest | 0.9187 | 0.6893 | 0.71 | 0.76 | 0.73 |
| Logistic Regression | 0.8823 | 0.5841 | 0.61 | 0.72 | 0.66 |
| Decision Tree (d=6) | 0.8241 | 0.4912 | 0.55 | 0.68 | 0.61 |

**Business Impact:** USD 150,000 estimated annual fraud savings vs. USD 43,000 total deployment cost = **304% Year 1 ROI**

---

## Repository Structure

```
insurtech-fraud-detection/
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Data loading, cleaning, EDA, distributions
│   ├── 02_Feature_Engineering.ipynb    # 10 engineered features, VIF analysis, PCA, t-SNE
│   ├── 03_Modeling.ipynb               # 4-model training, RandomizedSearchCV, comparison
│   └── 04_Evaluation_SHAP.ipynb        # SHAP explainability, PDP, bias & fairness audit
│
├── src/
│   └── fraud_detection_pipeline.py     # Complete end-to-end ML pipeline (standalone script)
│
├── deployment/
│   └── app.py                          # FastAPI inference endpoint (Step 8: Deployment)
│
├── reports/
│   ├── Mark_Villola_Capstone_Report.pdf
│   ├── Mark_Villola_Business_Presentation_withSpeakerNotes.pptx
│   └── Mark_Villola_Technical_Presentation_with SpeakerNotes.pptx
│
├── data/                               # Add raw data here (see setup instructions)
├── models/                             # Saved model artifacts generated on pipeline run
├── requirements.txt                    # All dependencies with pinned versions
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/AzzmanV/insurtech-fraud-detection.git
cd insurtech-fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

All library versions are pinned. Core stack: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `umap-learn`, `fastapi`, `uvicorn`.

### 3. Download the dataset

This project uses the **IEEE-CIS Fraud Detection** dataset (Vesta Corporation via Kaggle, 2019):

1. Go to: [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files in `data/raw/`

> **Note:** Dataset is not included in this repository due to Kaggle competition license terms. All feature engineering and model logic is fully reproducible once the data is downloaded.

### 4. Run the full pipeline

```bash
python src/fraud_detection_pipeline.py
```

This executes the complete ML lifecycle: data loading → preprocessing → feature engineering → model training → evaluation → SHAP analysis → saved artifacts in `models/`.

### 5. Run notebooks sequentially

```
01_EDA.ipynb
    ↓
02_Feature_Engineering.ipynb
    ↓
03_Modeling.ipynb
    ↓
04_Evaluation_SHAP.ipynb
```

All notebooks use `RANDOM_STATE = 42`. Running in sequence on the same data produces identical results.

---

## ML Pipeline Overview

### Dataset

| Attribute | Value |
|---|---|
| Source | Vesta Corporation / IEEE-CIS, Kaggle 2019 |
| Size | 590,540 transactions × 394 features |
| Target | `isFraud`: Binary (0 = Legitimate, 1 = Fraud) |
| Class distribution | 96.5% legitimate / 3.5% fraud |
| Adaptation | IEEE-CIS e-commerce features mapped to device insurance claim context |

### Feature Engineering (10 Domain-Specific Features)

| Feature | Construction | Fraud Signal |
|---|---|---|
| `log_claim_amt` | `np.log1p(TransactionAmt)` | Normalises right-skewed claim amounts |
| `days_since_policy` | `TransactionDT / 86400` | Claims < 14 days = 4× higher fraud rate |
| `is_early_claim` | Binary: 1 if `days_since_policy` < 14 | Direct timing manipulation flag |
| `claim_velocity` | Partner claims count, rolling 30-day | Batch fraud and repeat claimant detection |
| `email_risk_score` | Domain classification: 0 / 0.5 / 1.0 | Unverifiable identity at policy inception |
| `amt_to_device_ratio` | `TransactionAmt / category_device_avg` | Over-claiming: ratio > 1.2 is suspicious |
| `partner_fraud_rate_30d` | Rolling 30-day fraud rate by partner ID | Proactive channel risk monitoring |
| `mismatch_score` | Count of M1–M9 where value ≠ 'T' | Identity inconsistency across customer fields |
| `high_risk_email_x_early_claim` | `email_risk_score × is_early_claim` | Interaction term for compound risk |
| `log_dist1` | `np.log1p(dist1)` | Normalised claim velocity proxy |

### Class Imbalance Handling

- **SMOTE** oversampling applied on training fold only (inside `sklearn Pipeline` to prevent leakage)
- **`scale_pos_weight = 27.6`** (N_negative / N_positive) applied in XGBoost
- Evaluation uses **AUC-ROC** as primary metric (unaffected by imbalance) and **PR-AUC** as secondary

### Feature Selection

Three methods applied in sequence:
1. **VIF analysis** — dropped V131, V132, V133 (VIF > 10)
2. **XGBoost embedded importance** — dropped features with gain < 0.001
3. **SHAP post-hoc validation** — top 20 features account for >87% of model output variance

Final feature set: **55 features** from original 394.

### XGBoost Champion Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Diminishing returns beyond 500 |
| `max_depth` | 6 | Balances complexity and generalisation |
| `learning_rate` | 0.05 | Lower rate + more estimators = better generalisation |
| `subsample` | 0.8 | Row sampling per tree reduces overfitting |
| `colsample_bytree` | 0.7 | Feature sampling reduces tree correlation |
| `scale_pos_weight` | 27.6 | Compensates for class imbalance |
| `reg_alpha` | 0.1 | L1 regularisation |
| `reg_lambda` | 1.0 | L2 regularisation |
| `min_child_weight` | 5 | Prevents overfitting on small fraud clusters |

Tuned via `RandomizedSearchCV(n_iter=100, cv=5, scoring='roc_auc')`.

---

## Operational Decision Framework

The model outputs a continuous fraud probability score. Claims are routed to one of four tiers:

| Score Range | Operational Decision | Est. Volume | Precision | Recall |
|---|---|---|---|---|
| 0.00 – 0.20 | **Auto-approve** (Straight-Through Processing) | ~72% | — | — |
| 0.20 – 0.60 | **Standard adjuster review** | ~23% | ~62% | ~91% |
| 0.60 – 0.85 | **Enhanced review** (senior adjuster) | ~4% | ~76% | ~79% |
| > 0.85 | **Escalate + payment hold** | ~1% | ~89% | ~58% |

> The model never auto-rejects. It is an intelligent routing tool that directs claims to the right review level. All enhanced-review and escalated claims require human adjuster sign-off before any adverse action.

---

## Model Explainability (SHAP Analysis)

SHAP values computed on 5,000 stratified test records (2,500 fraud / 2,500 legitimate).

| Rank | Feature | Mean \|SHAP\| | Direction | Business Interpretation |
|---|---|---|---|---|
| 1 | V126 | 0.312 | High = lower risk | Device fingerprint consistency |
| 2 | V130 | 0.287 | High = lower risk | Session behavioural signal |
| 3 | `log_claim_amt` | 0.241 | High = higher risk | Large claims disproportionately fraudulent |
| 4 | `days_since_policy` | 0.198 | Low = higher risk | **Most actionable business signal** |
| 5 | V136 | 0.187 | See SHAP plot | Submission environment authenticity |
| 6 | `claim_velocity` | 0.156 | High = higher risk | Primary organised ring detection feature |
| 7 | `email_risk_score` | 0.143 | High = higher risk | Disposable email at policy activation |
| 8 | `card1` | 0.138 | Partner-specific | Individual partner fraud risk |
| 9 | `mismatch_score` | 0.121 | High = higher risk | Identity inconsistency at claim time |
| 10 | `amt_to_device_ratio` | 0.098 | > 1.0 = higher risk | Over-claiming detection |

Protected attributes (age, gender, geographic origin) do not appear in the top 30 SHAP contributors.

---

## Ethical AI & Bias Audit

Five fairness metrics evaluated across geographic regions, retail channels, and payment methods:

| Metric | Result | Threshold | Status |
|---|---|---|---|
| Demographic Parity (geographic) | 0.089 | < 0.10 | ✅ PASS |
| Equalized Odds — TPR (payment method) | 0.071 | < 0.10 | ✅ PASS |
| Equalized Odds — FPR (geographic) | 0.034 | < 0.10 | ✅ PASS |
| Disparate Impact Ratio | 0.847 | 0.80–1.20 | ⚠️ MONITOR |
| Predictive Parity (partner groups) | 0.052 | < 0.10 | ✅ PASS |

**Key decision:** `card4` (payment method, proxies socioeconomic status) excluded from the production model. AUC cost: 0.009 points (0.9412 → 0.9323). Fairness benefit exceeds the marginal performance cost.

**Human-in-the-loop guarantee:** No claim is automatically declined based on model score. All customers have a defined appeal path with plain-language SHAP explanation provided on request.

---

## Deployment — FastAPI Inference Endpoint (Step 8)

```bash
cd deployment
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Endpoint | Input | Output |
|---|---|---|---|
| `POST` | `/predict` | JSON claim features | `fraud_probability`, `risk_tier`, `top_3_shap_factors`, `recommended_action` |
| `GET` | `/health` | — | Model version, last trained date, current AUC-ROC |
| `GET` | `/explain/{claim_id}` | Claim ID | Full SHAP waterfall explanation |

### Example Response — `/predict`

```json
{
  "claim_id": "XC-20847",
  "fraud_probability": 0.91,
  "risk_tier": "ESCALATE",
  "recommended_action": "Hold payment. Escalate to fraud analyst.",
  "top_3_shap_factors": [
    {"feature": "days_since_policy", "value": 6, "impact": "HIGH RISK — claim filed 6 days after policy activation"},
    {"feature": "amt_to_device_ratio", "value": 1.37, "impact": "HIGH RISK — claim amount exceeds device category average"},
    {"feature": "partner_fraud_rate_30d", "value": 0.142, "impact": "HIGH RISK — partner 30-day fraud rate is 4x network average"}
  ]
}
```

---

## GenAI Integration — LLM-Powered Claims Triage (Step 9)

A Claude API integration converts SHAP model outputs into plain-language fraud alert narratives for claims adjusters. The LLM receives the top SHAP factors and directional impacts, then generates an actionable investigation brief.

**Sample output for a high-risk claim (Score: 0.91):**

> *"This claim shows three converging risk signals: (1) filed 6 days after policy activation — claims in the first 14 days have a 4× higher fraud probability; (2) claim amount of $823 exceeds the Water Damage category average of $600, suggesting possible over-claiming; (3) Retail Partner #4419 carries a rolling 30-day fraud rate of 14.2%, well above the network average of 3.5%. Recommended action: hold payment, escalate to fraud analyst, request purchase receipt and device serial verification."*

---

## Business Case

| Parameter | Value |
|---|---|
| Annual claims volume | ~5,400 claims/year |
| Estimated fraud rate | 8% |
| Annual fraudulent claims | ~432 |
| Model recall at 0.5 threshold | 79% → catches 341 of 432 |
| Fraudulent payouts prevented | ~USD 152,700/year |
| False positive review cost | ~USD 2,700/year |
| **Net annual fraud savings** | **~USD 150,000** |
| Model development cost (one-time) | USD 35,000 |
| Annual maintenance | USD 8,000 |
| **Year 1 ROI** | **304%** |
| **Year 2+ ROI** | **>1,600%** |

---

## Reproducibility Checklist

- [x] All notebooks use `RANDOM_STATE = 42`
- [x] SMOTE applied inside `Pipeline` on training fold only — zero leakage
- [x] 80/20 stratified train/test split preserves class ratio
- [x] 5-fold stratified cross-validation for all hyperparameter tuning
- [x] All dependencies pinned in `requirements.txt`
- [x] Run notebooks in sequence: `01 → 02 → 03 → 04`
- [x] Pipeline script produces identical results with fixed seed

---

## Limitations & Future Work

| Limitation | Impact | Planned Mitigation |
|---|---|---|
| IEEE-CIS dataset is e-commerce, not device insurance | Feature mappings are analogical | Retrain on ProTech proprietary claims data post-pilot |
| SMOTE tuned for 3.5% fraud rate vs. ProTech's ~8% | Threshold may need recalibration | Recalibrate after 3-month pilot data collection |
| V126/V130/V136 are opaque proprietary signals | Cannot be operationally interpreted | Replace with verified InsurTech behavioural features in production |
| No live concept drift monitoring | Model degrades over time | Implement PSI monitoring on top features; retrain annually |
| Cold start for new partners | `claim_velocity` and `partner_fraud_rate` default to zero | Apply 'new entity' flag; route all new partner claims to standard review for 90 days |

---

## Programme Context

This project was completed as the **Pillar 5 Capstone** for the AIM Postgraduate Diploma in Artificial Intelligence & Machine Learning. It demonstrates the end-to-end ML lifecycle applied to a real operational problem at ProTech Devices Asia — from business problem framing through data preprocessing, feature engineering, model training, explainability, bias auditing, and production deployment.

The pipeline developed here is the foundation for ProTech's planned fraud detection system deployment in H2 2026.

---

*Asian Institute of Management | School of Executive Education and Lifelong Learning*  
*Submitted: March 2026*
