## Customer Churn Prediction: A Comparative Analysis of Models with and without Sentiment Analysis :

## Overview of the Study
The core purpose of this study is to find the impact of Sentiment Analysis in predicting customer churn for the e-commerce industry by employing different predictive models. Furthermore, the study is also focused on observing which model is best in a more accurate prediction for determining the churn rate of customers.

## Process Involved in the Study
The whole study is divided into two phases:

* **In the first phase, all the relevant variables that are expected to be causing the customer churn are selected and then the predictive models are developed. In this, there will be no feedback from the customer is utilized.**
* **In the second phase, in addition to all the relevant variables obtained from EDA, the feedback provided by the customers is also included in this phase to extract the sentiment scores which are now added to the data frame. Again, the churn predictive models are developed with this data.**

*   Finally, the metrics from both these phases will be reviewed and interpreted to understand if the inclusion of the sentiment analysis will be helpful for the organization in better understanding why their customers are parting away without making any future transactions with the organization.
*   The effectiveness of these models, with and without sentiment analysis, is compared to understand if sentiment analysis aids in better predicting why customers may discontinue transactions with the organization.

## Research Question
"How does sentiment analysis impact predicting the customer churn of an organization?"

## Flowchart of process involved in the Study
<img width="796" alt="image" src="https://github.com/GaneshKotaSLU/Customer-Churn-Prediction/assets/145741973/a0fcbaee-a8dd-4ef8-9355-8b02559536f7">

## Checking the Classes Proportion (Class Imbalance)

* 1 - Represented the churned customer
* 0- Represents the non-churned customer
<img width="625" alt="image" src="https://github.com/GaneshKotaSLU/Customer-Churn-Prediction/assets/145741973/a70b11c6-c999-4afb-a55b-736753e83dc8">


### Sentiment Analysis Integration
* **DistilBERT** is used for performing the sentiment analysis.
- **Hugging Face API Connection:**
  ```python
  from huggingface_hub import notebook_login
  notebook_login()
  ```




### IMDB Dataset Pre-training (fine-tuning)
 ```python
from datasets import load_dataset
imdb = load_dataset("imdb")
```
### Model Performance Metrics
#### Cross-Validation Scores:
Accuracy, Recall, Precision, F1 Score, ROC_AUC.

Random Forest Model Metrics:

```python
import pandas as pd
# DataFrame setup
rf_output = pd.DataFrame({
    'Training': [train_cv_acc_rf, train_cv_recall_rf, train_cv_precision_rf, train_cv_f1_rf, roc_auc_train_rf],
    'Testing': [test_cv_acc_rf, test_cv_recall_rf, test_cv_precision_rf, test_cv_f1_rf, roc_auc_test_rf]
}, index=['Accuracy', 'Recall', 'Precision', 'F1', 'ROC_AUC'])
print(rf_output)
```

### Feature Importance Analysis
#### Variable Importances for Random Forest and LightGBM:
### Plot feature importance
```python

import matplotlib.pyplot as plt

plt.barh(sorted_feature_names, sorted_feature_importances)
plt.xlabel('Feature Importance')
plt.title('Variable Importance for Random Forest')
plt.show()
```

## Next Steps
This prototype needs to be integrated to the real-time data to predict the customer churn behavior.

### Support

Support our work by starring our GitHub repository.

## Conclusion
This phase of the study explores how incorporating customer sentiment analysis affects churn prediction accuracy across different models. The findings from this phase will guide future research directions and potential improvements in customer retention strategies.


This README provides a concise overview of the project, describes the processes involved, outlines the improvements made in the current phase, and presents the future steps. It is formatted to be clear and accessible for other researchers or stakeholders who might review the project on platforms like GitHub.
