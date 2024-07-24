# Customer Churn Prediction: A Comparative Analysis of Models with and without Sentiment Analysis

## Table of Contents
- [Overview](#overview-of-the-study)
- [Process](#process-involved-in-the-study)
- [Research Question](#research-question)
- [Flowchart](#flowchart-of-process-involved-in-the-study)
- [Class Imbalance](#checking-the-classes-proportion-class-imbalance)
- [Sentiment Analysis Integration](#sentiment-analysis-integration)
- [Model Performance Metrics](#model-performance-metrics)
- [Feature Importance Analysis](#feature-importance-analysis)
- [Installation](#installation)
- [Results](#results)
- [Challenges and Limitations](#challenges-and-limitations)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Technologies Used](#technologies-used)
- [Acknowledgments](#acknowledgments)
- [Next Steps](#next-steps)
- [Support](#support)

## Overview of the Study
The core purpose of this study is to find the impact of Sentiment Analysis in predicting customer churn for the e-commerce industry by employing different predictive models. Furthermore, the study is also focused on observing which model is best in a more accurate prediction for determining the churn rate of customers.

## Process Involved in the Study
The whole study is divided into two phases:

1. In the first phase, all the relevant variables that are expected to be causing the customer churn are selected and then the predictive models are developed. In this, there will be no feedback from the customer is utilized.
2. In the second phase, in addition to all the relevant variables obtained from EDA, the feedback provided by the customers is also included in this phase to extract the sentiment scores which are now added to the data frame. Again, the churn predictive models are developed with this data.

Finally, the metrics from both these phases will be reviewed and interpreted to understand if the inclusion of the sentiment analysis will be helpful for the organization in better understanding why their customers are parting away without making any future transactions with the organization.

The effectiveness of these models, with and without sentiment analysis, is compared to understand if sentiment analysis aids in better predicting why customers may discontinue transactions with the organization.

## Research Question
"How does sentiment analysis impact predicting the customer churn of an organization?"

## Flowchart of process involved in the Study
![Flowchart](https://github.com/GaneshKotaSLU/Customer-Churn-Prediction/assets/145741973/a0fcbaee-a8dd-4ef8-9355-8b02559536f7)

## Checking the Classes Proportion (Class Imbalance)
- 1 - Represented the churned customer
- 0 - Represents the non-churned customer

![Class Imbalance](https://github.com/GaneshKotaSLU/Customer-Churn-Prediction/assets/145741973/a70b11c6-c999-4afb-a55b-736753e83dc8)

## Sentiment Analysis Integration
DistilBERT is used for performing the sentiment analysis. DistilBERT is a smaller, faster, cheaper and lighter version of BERT that still retains a lot of BERT's language understanding capabilities.

### Hugging Face API Connection:
```python```
``` sh 
from huggingface_hub import notebook_login
notebook_login()
```

### IMDB Dataset Pre-training (fine-tuning)
``` sh
from datasets import load_dataset
imdb = load_dataset("imdb")
```

### Model Performance Metrics
- Cross-Validation Metrics
  We evaluate our models using the following metrics:
  1. Accuracy
  2. Recall
  3. Precision
  4. F1 Score
  5. ROC_AUC
- Random Forest Model Metrics:
  ``` sh
  import pandas as pd
  ## Dataframe setup
  rf_output = pd.DataFrame({
  'Training': [train_cv_acc_rf, train_cv_recall_rf, train_cv_precision_rf, train_cv_f1_rf, roc_auc_train_rf],
  'Testing': [test_cv_acc_rf, test_cv_recall_rf, test_cv_precision_rf, test_cv_f1_rf, roc_auc_test_rf]
  }, index=['Accuracy', 'Recall', 'Precision', 'F1', 'ROC_AUC'])
  print(rf_output)
  ```
### Feature Importance Analysis
We analyze the importance of different features in our models to understand which factors contribute most to customer churn.
- Variable Importances for Random Forest and LightGBM:
  ``` sh
  import matplotlib.pyplot as plt
  plt.barh(sorted_feature_names, sorted_feature_importances)
  plt.xlabel('Feature Importance')
  plt.title('Variable Importance for Random Forest')
  plt.show()
  ```
### Installation
To set up the project environment:

1. Clone the repository:
   ``` sh
   git clone https://github.com/GaneshKotaSLU/Customer-Churn-Prediction.git
   ```
2. Navigate to the Project Directory:
   ``` sh
   cd Customer-Churn-Prediction
   ```

### Results
Our analysis shows that incorporating sentiment analysis into churn prediction models can significantly improve their accuracy. Key findings include:

* Models with sentiment analysis outperformed traditional models atmost 3%.
* Customer feedback sentiment was found to be a strong predictor of churn.
* The SVM and Random Forest model showed the best overall performance.

## Challenges and Limitations

* Data quality and completeness varied across different customer segments.
* The sentiment analysis model may not capture all nuances in customer feedback.
* The current approach doesn't account for time-series aspects of customer behavior.

## Contributing
Welcome contributions to this project. Please follow these steps:

## Fork the repository
- Create a new branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citation
If you use this work in your research, please cite:
``` sh
Kota, G. (2023). Customer Churn Prediction: A Comparative Analysis of Models with and without Sentiment Analysis. GitHub repository, https://github.com/GaneshKotaSLU/Customer-Churn-Prediction
```
## Technologies Used
The below are few of the technologies used in this project.
* Python 3.8+
* Pandas
* Scikit-learn
* TensorFlow
* Hugging Face Transformers
* Matplotlib
* LightGBM
## Acknowledgments

Thanks to the Hugging Face team for their excellent NLP tools and models.
This project was inspired by recent advancements in sentiment analysis and its applications in business intelligence.

## Next Steps
This prototype needs to be integrated with real-time data to predict customer churn behavior. Future work includes:

## Implementing real-time data integration
- Exploring additional machine learning models, such as neural networks.
- Conducting A/B testing in a production environment.
- Investigating the impact of external factors (e.g., market trends, competitor actions) on churn rates.

## Support
Support our work by starring our GitHub repository. For any questions or suggestions, please open an issue in the repository.

``` sh
This comprehensive README provides a detailed overview of your project, its methodology, results, and future directions. It includes all the sections we discussed earlier, with placeholders for specific results and findings that you can fill in with your actual data. The structure is designed to be informative for both technical and non-technical readers, making your project more accessible and encouraging collaboration.
```
