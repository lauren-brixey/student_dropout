# Predicting Student Dropout Using Supervised Learning

## Project Overview  
This project applies **supervised machine learning** to predict whether a student is at risk of dropping out based on demographic, engagement, and performance data.  

Student retention is a critical challenge in higher education. High dropout rates can lead to:  
- Financial losses for institutions  
- Reputational challenges  
- Negative academic outcomes  

By detecting early warning signs of dropout, institutions can take **proactive measures** (academic support, engagement programs, financial aid) to improve both retention and student success.  

---

## Business Problem  
Educational institutions need to identify **at-risk students** as early as possible in order to:  
- Allocate support resources efficiently  
- Reduce the likelihood of students leaving  
- Improve student outcomes and institutional performance  

A data-driven predictive model allows institutions to shift from **reactive** to **proactive** retention strategies.  

---

## Project Objectives  
- Explore and preprocess student data across three stages of the academic journey  
- Engineer meaningful features (demographics, engagement, performance)  
- Train and evaluate supervised models (**XGBoost** and **Neural Networks**)  
- Optimise for **recall** (minimising false negatives, ensuring at-risk students are flagged)  
- Interpret model outputs to provide actionable insights for retention strategy  

---

## Datasets  
The analysis uses three datasets representing different stages of a student’s academic journey:  

1. **Stage 1: Early**  
   - Demographic & background features only  
2. **Stage 2: Mid**  
   - Stage 1 features + engagement data (`AuthorisedAbsenceCount`, `UnauthorisedAbsenceCount`)  
3. **Stage 3: Late**  
   - Stage 1 & 2 features + academic performance (`AssessedModules`, `FailedModules`, `PassedModules`)  

---

## Methodology  

### 1. Data Preprocessing  
- Dropped features with >50% missing values  
- Median imputation for <20% missing values  
- One-hot encoding (nominal), label encoding (ordinal)  
- Standardisation for neural networks  
- Engineered features: e.g. `Age` from `DateOfBirth`  
- Target variable transformed to binary `DroppedOut` (1 = dropout, 0 = completed)  

### 2. Models  
- **XGBoost (XGBClassifier)** with `scale_pos_weight` to address class imbalance  
- **Neural Network (Keras Sequential)**  
  - 2 hidden layers (ReLU/Tanh activations)  
  - Sigmoid output for binary classification  
  - Class weights, early stopping, L2 regularisation  

### 3. Hyperparameter Tuning  
- **XGBoost:** grid search on `learning_rate`, `max_depth`, `n_estimators`  
- **Neural Network:** nested loop search over `optimizer`, `activation`, `neurons`, `batch_size`  

### 4. Evaluation Metrics  
- **Primary metric:** Recall (minimise false negatives)  
- **Supporting metrics:** Precision, F1-score, ROC-AUC  
- Accuracy considered only if class balance permits  

---

## Results  

- **Feature quality** (Stage progression) had a greater impact than hyperparameter tuning.  
- **Stage 1 (Demographics only):** modest recall (~0.77), limited predictive power  
- **Stage 2 (Engagement features):** improved recall, precision decreased slightly  
- **Stage 3 (Performance features):** highest recall and overall performance  

**Best Model:**  
- **Tuned XGBoost (Stage 3)**  
  - Recall: **0.925**  
  - Precision: **0.842**  
  - F1: **0.882**  
  - Correctly identified **695 at-risk students**, missing only **56 cases**  

---

## Conclusions & Recommendations  
- Strong predictors (academic performance features) are key to effective dropout prediction  
- Recall-focused models ensure **at-risk students are rarely missed**  
- **XGBoost (Stage 3 tuned)** is the most reliable candidate for deployment  
- Future work:  
  - Deploy model into student monitoring dashboards  
  - Test real-time intervention strategies  
  - Expand features (financial aid, extracurricular involvement, etc.)  

---

## Tech Stack  
- **Languages:** Python  
- **Libraries:** scikit-learn, XGBoost, Keras, TensorFlow, Pandas, NumPy, Matplotlib, Seaborn  
- **Methods:** Supervised Learning, Class Imbalance Handling, Hyperparameter Tuning, Model Evaluation  

---

## Repository Structure  
```
├── student_dropout.ipynb # Jupyter notebook with full analysis
├── student_dropout.pdf # Stakeholder report 
├── README.md # Project documentation
```
