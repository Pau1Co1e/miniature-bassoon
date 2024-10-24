# **Titanic Survival Prediction: Midterm Project Report**

## **1. Title and Introduction**
This project aims to predict whether a passenger survived or perished during the Titanic disaster using machine learning techniques. The dataset used for this project is sourced from the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic), which includes passenger information such as age, gender, class, and other features.

---

## **2. Problem Statement and Background**

**Problem Definition**: 
The goal is to predict whether a passenger survived the Titanic disaster (binary classification: 0 for deceased, 1 for survived) based on various features like age, gender, ticket class, family size, etc.

**Real-World Context**: 
Predictive modeling of survival can offer insights into which factors played the most significant roles in determining survival during the disaster. Understanding these factors is useful not just for historical analysis but also as an exercise in applying supervised learning to imbalanced datasets.

---

## **3. Data Description and EDA**

**Dataset**: 
The Titanic dataset consists of two files: `train.csv` for training the models and `test.csv` for generating predictions. Key features include:
- **PassengerId**: Unique identifier for each passenger
- **Pclass**: Passenger’s class (1st, 2nd, 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard the Titanic
- **Parch**: Number of parents/children aboard the Titanic
- **Fare**: Fare paid for the ticket
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

**Exploratory Data Analysis (EDA)**:
- **Survival by Gender**: Women had a significantly higher survival rate than men.
- **Survival by Class**: Passengers in 1st class had a much higher survival rate compared to those in 3rd class.
- **Survival by Age**: Younger passengers had a higher likelihood of survival.
- **Family Size**: Larger families had a slightly lower survival rate, while individuals and small families had better chances.
- **Embarkation Port**: Passengers who embarked from Cherbourg had a higher survival rate compared to those who embarked from Southampton.

---

## **4. Methods: Preprocessing and Feature Engineering**

**Preprocessing**:
- **Handling Missing Values**: 
  - Age: Missing values were filled with the median age.
  - Embarked: Missing values were replaced with the most common port (mode).
  - Fare: Missing values were filled with the median fare.
- **Feature Engineering**:
  - **Family Size**: Created a new feature by summing `SibSp` and `Parch` to capture family size aboard the Titanic.
  - **Title Extraction**: Extracted titles (e.g., Mr., Mrs., Miss) from the `Name` feature for better representation of passenger categories.
  - **CabinAvailable**: Created a binary feature to indicate whether cabin information was available.
- **Encoding Categorical Variables**: One-hot encoding was applied to categorical features such as `Sex`, `Embarked`, and `Title`.
- **Scaling Continuous Features**: Age and Fare were standardized using `StandardScaler`.

---

## **5. Models and Hyperparameter Tuning**

**Logistic Regression**:
- **Parameters tuned**: `C` (regularization strength) and `solver` (optimization algorithm).
- **Best parameters**: C = 1, solver = 'liblinear'

**Random Forest**:
- **Parameters tuned**: `n_estimators`, `max_depth`, `min_samples_split`.
- **Best parameters**: n_estimators = 200, max_depth = 20, min_samples_split = 5

**Support Vector Machine (SVM)**:
- **Parameters tuned**: `C`, `kernel`, and `gamma`.
- **Best parameters**: C = 1, kernel = 'rbf', gamma = 'scale'

**Cross-Validation Results**:
- Logistic Regression: **Accuracy** = 0.8305
- Random Forest: **Accuracy** = 0.8036
- SVM: **Accuracy** = 0.8316

---

## **6. Evaluation and Results**

**Final Model Selection**: 
The Support Vector Machine (SVM) model was selected as the final model due to its superior cross-validation accuracy of **0.8316**.

**Test Set Predictions**: 
After selecting the best model (SVM), predictions were generated on the test set, which were formatted and submitted for evaluation on Kaggle.

---

## **7. Additional Contribution: Hyperparameter Tuning and Model Comparison**

In addition to basic model training, hyperparameter tuning was applied to optimize each model’s performance. The grid search was used to find the optimal parameters for Logistic Regression, Random Forest, and SVM models.

---

## **8. Conclusion**

- **Key Insights**: The analysis confirms that women, children, and first-class passengers had a higher chance of survival. Age and family size
