# Airline Loyalty Customer Churn

**Business question:** Which airline loyalty members are most likely to cancel in the near future, and which high-value customers should the retention team contact first?

The objective is not only to predict churn, but to help prioritize limited retention offers for members with the highest churn risk and customer value.

This is a retrospective portfolio project using historical cancellation labels. Model metrics and ROI estimates are illustrative, not live airline results.

## Approach

- **Snapshot design to avoid target leakage.** Flight activity covers 2017–2018, but cancellations go back to 2013, and members record zero flights after cancelling. Totalling activity for every cancelled member would teach the model that people who already left stopped flying. Instead, the notebook takes a snapshot on **31 Dec 2017**. It uses only 2017 activity for the **12,305 members active at that date** and predicts who cancels during **2018** (3.9% churn rate).
- **Features:** 2017 flights, distance, points, redemptions, months flown, recency (months since last flight), half-year trend, Q4 activity, membership tenure, and profile fields.
- **Models:** a do-nothing baseline, Logistic Regression, KNN, Random Forest, and SVM. They are compared with 5-fold stratified cross-validation on the training set only, ranked by **PR-AUC** (the right metric for a rare outcome).
- **Threshold:** chosen from out-of-fold training predictions by maximizing F2 (recall-weighted), never from the test set.
- **Final check:** one evaluation on a 20% stratified hold-out set, with a confusion matrix, ROC and precision-recall curves, and a decile lift table.

## Findings

- After removing leakage, **Random Forest** ranks best (hold-out ROC-AUC **0.71**, PR-AUC **0.10** vs 0.04 for random). The earlier ROC-AUC of 0.87 came from members who had already cancelled.
- The **top 20%** of ranked members contain about **half of all 2018 churners** (2.5× random).
- **Tenure** is the strongest signal: first-year members churn at about 10%, against about 3% for longer-tenured members. The lowest quarter of 2017 flyers churn at about 7%, against about 3% for everyone else.
- At the chosen threshold (0.36), the model flags 352 hold-out members and catches **44%** of churners at **12%** precision. Accuracy is not reported as a headline: predicting "nobody churns" scores 96%.

## Business simulation

The simulation ranks members by **expected value at risk = churn probability × CLV** and compares campaigns against random outreach. The assumptions are illustrative: a CAD 50 offer, 20% success, and 10% of CLV retained. Model-targeted campaigns reach 2–4× more churners (and 3–4× more CLV) than random outreach. However, a CAD 50 offer only breaks even at about a 39% success rate for the top value-at-risk group. That points to a cheaper first touch, or a pilot to measure the real success rate.

## Run the notebook

1. Open [`notebook/Airline_Loyalty_Program.ipynb`](./notebook/Airline_Loyalty_Program.ipynb) in Jupyter or Google Colab.
2. Install the packages: `pip install -r requirements.txt`.
3. Run all notebook cells in order. The source CSV files are in [`DATA/`](./DATA/).
4. The notebook saves hold-out risk scores to [`outputs/airline_churn_predictions.csv`](./outputs/airline_churn_predictions.csv).

The source dataset is from Maven Analytics' Airline Loyalty Program dataset.

<img width="1199" height="1656" alt="image" src="https://github.com/user-attachments/assets/59a72b36-5a3f-4451-9baa-3d1608779930" />

