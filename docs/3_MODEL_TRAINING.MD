# ML Model Training

> This segment of the documentation acts as a plan which might be changed as per our requirements as we progress.

## Understanding the Training Process

To predict **CG (Cumulative Grade)** based on the 20 identified factors, we followed a systematic approach to train our machine learning model. The process involved:

1. **Preparing the data**
2. **Training the model**
3. **Evaluating its performance**

---

## Data Preparation

The first step was to prepare the data so the model could understand it:

- **Handling Missing Data**:
    - Filled missing numerical values (e.g., study hours) with some data.
    - Used the most common value for categorical data (e.g., branch or school type).

- **Scaling the Data**:
    - Scaled features with different ranges (e.g., attendance, previous scores) to ensure uniformity.

- **Splitting Data**:
    - Divided the dataset into (Planned):
        - **70% Training Set**
        - **30% Testing Set**

---
## Choosing the Right Model

Multiple machine learning algorithms were tested to identify the best fit for this problem:

| **Algorithm**            | **Description**                                                                 |
|---------------------------|---------------------------------------------------------------------------------|
| **Linear Regression**     | A starting point to understand basic relationships between factors and CG.     |
| **Random Forest Regressor** | Advanced model for handling complex, non-linear relationships.                 |
| **Gradient Boosting**     | Known for high accuracy by learning from its own mistakes.                     |
| **Support Vector Machines (SVM)** | Suitable for high-dimensional, less structured data.                          |

### Observations
- **Random Forest** and **Gradient Boosting** emerged as the most balanced in terms of accuracy and interpretability.

---

## Preliminary Observations

During the initial phase:
- Strong correlation observed between **study hours**, **previous scores**, and **CG**.
- Behavioral factor like  **extracurricular participation** significantly influenced academic performance.

---

## Training the Model

### Random Forest Training
- Adjusted key parameters such as:
    - Number of trees (**n_estimators**)
    - Tree depth
- Ensured the model didn’t overfit the data.

### Cross-Validation
- Used cross-validation to evaluate performance on new data by splitting the data into multiple subsets.

---
