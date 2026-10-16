# Bank Transaction Anomaly Detection & Customer Fund Value — Analysis Plan

**Business direction**: Amazon seller / payment-institution risk control · Revenue Protection. The first three projects (procurement, AR, e-commerce) quantified "revenue reconciliation gaps"; this project uses 1.048M rows of real personal bank transactions to identify **anomalous transactions and fund risk**, stopping fraud/anomaly-driven fund loss, and tiering customers by fund value.

## 1. Analysis Purpose
- **Revenue/fund protection**: with high daily volume, use unsupervised methods to identify **anomalous transactions** (extreme large amounts, abnormal frequency, balance-transaction mismatch, late-night transactions), review them first, and prevent fund loss from fraud / money laundering / operational errors.
- **Customer fund-value tiering**: identify high-value accounts by balance/amount/frequency, guiding differentiated service and risk-control resource allocation.
- **Honest boundary**: each customer averages only 1.19 transactions and balance vs transaction has no correlation (r=0.06) → **a true balance reconciliation cannot be done**. This project is positioned as **transaction-level anomaly detection**, not a fake "account reconciliation".

## 2. Analysis Metrics (definitions)
- Transaction amount: distribution, median, P90/P99/P999, heavy tail (max ₹1.56M).
- Per-customer transaction-count distribution (most have 1).
- Balance vs transaction gap (share and amount where `balance < transaction`).
- Time-of-day distribution: late-night (00:00–05:59) transaction share.
- City/gender distribution.
- Anomaly score Top-N queue (amount to review).
- Customer fund-value tiers (four levels).

## 3. Cleaning & Preprocessing (steps + definitions)
- Date parsing: `1/8/16` two-digit year → `2016-01-08` (consistent century inference).
- Time: `HHMMSS` integer → `datetime.time` / hour bucket.
- Gender: missing (1,100) / `T` (1) treated as `unknown`.
- Amount: confirm no negatives (all positive), 0 values 0.08%.
- Dedup; build **transaction-level table** (1.048M rows) + **customer-level table** (884K customer aggregates).

## 4. Modeling (ML + statistics; explain principle/why/how)
- **Statistical baseline**: Modified Z-score / IQR outliers (interpretable, easy to deploy).
- **Unsupervised anomaly detection**: Isolation Forest + DBSCAN (compare their principles; why they fit unlabeled data).
- **Customer feature modeling**: balance/amount/frequency/time-of-day → customer risk score + fund-value tiers.
- **Deep-learning judgment (honest)**: 1.048M transaction rows could train an autoencoder, but in anomaly detection **interpretability > precision** (must answer "why this line is flagged"), and unlabeled data cannot be trained supervised → **DL not used**, interpretable tree ensembles / density methods used instead, with the reason stated (consistent with the user's "use only where the business needs it, explain the principle" preference).

## 5. Visualizations (each chart Insight + Action)
- Amount distribution (log-scale heavy tail).
- City transaction-volume top comparison.
- Time-of-day heatmap (late-night anomalies).
- Anomaly score top queue bars.
- Balance-gap distribution.
- Customer fund-value tiers.

## 6. Recommendations (landing)
- Anomalous transaction review queue (Top-N).
- Late-night large-amount + balance-mismatch real-time monitoring rules.
- Differentiated service for high-value accounts.
- Adding external fraud labels enables supervised learning (next step).

## Deliverables
- `scripts/` cleaning/analysis/modeling/diagnostics.
- `data/output/` parquet + metrics + ml_result.
- `tests/` definition tests.
- `report/` bilingual HTML report (forest-green + amber palette, each chart Insight+Action).
- `README.md` (English, for job applications).
