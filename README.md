# Airline Loyalty Customer Churn

**Business question:** Which airline loyalty members are most likely to cancel in the near future, and which high-value customers should the retention team contact first?

The objective is not only to predict churn, but to help prioritize limited retention offers for members with the highest churn risk and customer value.

This is a retrospective portfolio project using historical cancellation labels. Model metrics and ROI estimates are illustrative, not live airline results.

## Findings

- Logistic Regression is recommended when the priority is to catch potential churners: it achieved **80.9% recall** and **0.875 ROC-AUC** on the hold-out set at the default 0.50 threshold.
- Members below the median in total distance had **22.8% observed churn**, compared with **1.9%** for members above the median. Prioritize lower-activity members, especially those with high customer lifetime value (CLV).
- The notebook assigns Low, Medium, and High risk tiers, with a suggested retention action for each.
- Lowering the decision threshold from 0.50 to 0.35 flagged **1,178** rather than **919** members. Recall increased from **80.9% to 88.4%**, while precision decreased from **36.3% to 31.0%**. Choose a threshold based on outreach capacity and offer cost.

## Business simulation

An illustrative high-risk campaign assumes ₹500 annual customer value, ₹50 retention cost per contact, and a 20% retention success rate. Under these assumptions, the hold-out simulation estimates a **₹5,100 net loss**; the break-even success rate is **23.7%**. These are assumptions, not actual airline costs, revenue, or proven campaign impact.

## Run the notebook

1. Open [`notebook/Airline_Loyalty_Program.ipynb`](./notebook/Airline_Loyalty_Program.ipynb) in Jupyter or Google Colab.
2. Install the packages listed in [`requirement.txt`](./requirement.txt), if needed.
3. Run all notebook cells in order. The source CSV files are in [`DATA/`](./DATA/).
4. The notebook saves hold-out risk scores to [`outputs/airline_churn_predictions.csv`](./outputs/airline_churn_predictions.csv).

The source dataset is from Maven Analytics' Airline Loyalty Program dataset.
