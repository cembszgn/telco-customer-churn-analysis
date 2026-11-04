# Telco Customer Churn Analysis

## 1. Project Overview

This project focuses on analyzing and predicting customer churn in the telecommunications industry using data mining and machine learning techniques.

Customer churn refers to customers discontinuing their services or leaving a company. Predicting potential customer loss is important for businesses because identifying customers who are likely to leave can help companies develop customer retention strategies.

The main objective of this project is to analyze customer characteristics, identify patterns associated with churn, and develop classification models to predict customer churn.

## 2. Dataset

The dataset was obtained from Kaggle.

- **Dataset Name:** Telco Customer Churn
- **Author:** blastchar
- **Source:** [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Number of Observations:** 7,043
- **Number of Variables:** 21
- **Target Variable:** Churn

The dataset contains customer demographics, subscribed services, contract information, payment details, and customer churn status.

## 3. Technologies and Libraries

The project was developed using R programming language.

The following packages were used:

| Technology / Library | Purpose |
|---|---|
| R Programming | Data analysis and model development |
| Tidyverse | Data manipulation and preprocessing |
| Caret | Data preparation and model evaluation |
| Corrplot | Correlation analysis and visualization |
| e1071 | Support Vector Machine implementation |
| class | K-Nearest Neighbors implementation |
| R Markdown | Project documentation and reporting |

## 4. Project Workflow

The project follows several stages of the data mining process.

### 4.1 Data Exploration
- Dataset structure and variable types were examined.
- Descriptive statistics were analyzed.
- Customer churn distribution was investigated.

### 4.2 Data Visualization
- Churn distribution analysis
- Contract type and churn relationship
- Monthly charges distribution
- Boxplot analysis of monthly charges by churn status
- Tenure and monthly charges relationship

### 4.3 Data Preprocessing
- Removal of unnecessary variables
- Data type conversion
- Missing value analysis
- Missing value imputation using median
- Categorical variable transformation
- Outlier analysis
- Consistency checks
- Correlation analysis
- Dummy variable transformation
- Data standardization

### 4.4 Model Development

Two classification algorithms were implemented:

**Support Vector Machine (SVM)**
- Radial kernel
- Customer churn classification

**K-Nearest Neighbors (KNN)**
- Distance-based classification
- Number of neighbors (k): 5

The dataset was divided into:
- Training Set: 70%
- Testing Set: 30%

## 5. Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix
- Sensitivity
- Specificity

### Model Performance Comparison

| Model | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| SVM | 80.87% | 68.66% | 51.25% |
| KNN | 75.85% | 54.65% | 52.50% |

### Results

Based on the evaluation results, SVM achieved higher overall accuracy and precision compared to KNN.

KNN achieved a slightly higher recall score, indicating a marginal advantage in identifying customers who actually churned.

The results demonstrate the importance of evaluating classification models using multiple performance metrics rather than relying solely on accuracy.

## 6. Business Relevance

Customer churn prediction can support business decision-making and customer retention strategies.

Potential applications include:

- Identifying customers with a high risk of leaving.
- Developing targeted customer retention campaigns.
- Evaluating the relationship between contract types and customer churn.
- Supporting pricing and customer service strategies.

## 7. Project Files

The repository contains the following project materials:

- **README.md:** Project overview and documentation.
- **R Markdown (.Rmd):** Source code and analysis workflow.
- **PDF Report:** Detailed project report and model evaluation results.
- **Dataset:** Original dataset source is provided through Kaggle.

## 8. How to Run the Project

1. Install R and RStudio.
2. Download the dataset from the Kaggle source provided above.
3. Place the CSV file in the same directory as the R Markdown file.
4. Install the required R packages:

```r
install.packages(c("tidyverse", "caret", "corrplot", "e1071", "class"))
```

5. Open the R Markdown file in RStudio.
6. Run the analysis or knit the document to generate the report.

## 9. Conclusion

This project applies data mining and machine learning techniques to investigate customer churn in the telecommunications industry.

Through data exploration, preprocessing, visualization, and classification modeling, two algorithms (SVM and KNN) were implemented and evaluated.

The comparison showed that SVM achieved better overall accuracy, while KNN produced a slightly higher recall score.

The project demonstrates the application of data analysis and machine learning methods to a practical business problem.

---

**Project Category:** Data Mining / Machine Learning  
**Programming Language:** R  
**Documentation:** R Markdown  
**Dataset Source:** Kaggle – Telco Customer Churn
