# Bankruptcy Prediction Project

This project predicts company bankruptcy based on financial ratios using a random forest model. 

The dataset contains financial information and bankruptcy status of approximately 7000 Taiwanese companies that were listed on the Taiwan Stock Exchange from 1999-2009. The dataset is obtained from Kaggle: [Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction).

## Files

- `bankruptcy_data.csv`: Dataset containing financial ratios and bankruptcy status of companies
- `project_proposal.pdf`: Information about the project proposal
- `project_report.pdf`: Project write-up
- `randformForest_bankruptcy_prediction.ipynb`: Jupyter Notebook containing Python code for building random forest models for bankruptcy prediction

   1. **Data Exploration**
      - Identified imbalanced data with only 2-3% of companies being bankrupt.
   
   2. **Data Wrangling**
      - Removed "Liability-Assets Flag" and "Net Income Flag" categorical columns as their meanings were unclear.
   
   3. **Model Building**
      - Conducted principled training/test data splitting adhering to a 75/25 ratio for rigorous model evaluation.
      - Addressed class imbalance issue using Synthetic Minority Oversampling Technique (SMOTE) to ensure equitable representation.
      - Employed Principal Component Analysis (PCA) and Kernel PCA to address interdependencies among financial data and reduce dimensionality.
      - Built random forest models to capture nonlinearity effectively in the data.
   
   4. **Model Evaluation**
      - Prioritized recall rate in model performance evaluation to enhance bankruptcy detection effectiveness and minimize false negatives.
      - Utilized confusion matrix and Precision-Recall Curve (PR-AUC) for comprehensive evaluation of model predictive capacity and discriminatory ability.
      - After applying SMOTE to address class imbalance, we constructed several random forest models using different combinations of PCA and kernel methods.
     
**Note:** Other models such as logistic regression and KNN were also explored by other members of the project team, but their results are not included in this README for brevity. This was a course project for GeorgiaTech's CDA course in Summer 2023.

## Takeaways

Among these models, the Random Forest (w/SMOTE) model achieved the highest area under the curve for precision-recall (AUC-PR) with a value of 0.33. This suggests that the random forest approach may not be the most suitable model for predicting bankruptcy for this dataset.
