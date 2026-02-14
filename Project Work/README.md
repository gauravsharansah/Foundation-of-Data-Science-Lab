# Employee Attrition Prediction using Logistic Regression

A comprehensive machine learning project that predicts employee attrition using logistic regression implemented from scratch with NumPy. This project includes extensive exploratory data analysis (EDA), data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Features](#features)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project aims to predict whether an employee will leave the company (attrition) based on various factors such as age, salary, experience, department, and performance metrics. The model is built using logistic regression with gradient descent, implemented from scratch without using sklearn's LogisticRegression model.

**Key Highlights:**
- Custom implementation of logistic regression with gradient descent
- Comprehensive EDA with 15+ visualizations
- Feature engineering and data preprocessing
- Multiple evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Handles missing data and categorical variables

## Dataset

The dataset (`employee_data.csv`) contains 1000 employee records with the following features:

### Features:
- **ID**: Unique employee identifier
- **Age**: Employee age (18-65 years)
- **Gender**: Male/Female/Other
- **Education**: High School/Bachelor's/Master's/PhD
- **Salary**: Annual salary ($3,210 - $149,820)
- **Experience**: Years of work experience (0-48 years)
- **Department**: IT/Finance/HR/Marketing/Sales
- **Remote_Work**: Yes/No
- **Joining_Year**: Year the employee joined (1990-2024)
- **Performance_Score**: Rating from 1-10
- **Certifications**: Number of professional certifications (0-10)
- **Attrition**: Target variable (Yes/No)

### Data Quality:
- Total records: 1000
- Missing values present in: Age, Gender, Education, Experience, Joining_Year, Certifications
- Missing value percentages range from 3% to 10%

## Project Structure

```
.
├── employee_data.csv           # Dataset file
├── main2.ipynb                 # Main Jupyter notebook
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd employee-attrition-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `main2.ipynb`

3. Run all cells sequentially (Kernel → Restart & Run All)

### Quick Start Example

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("employee_data.csv")

# The notebook contains all preprocessing and model training code
# Simply run the cells in order to:
# 1. Perform EDA
# 2. Preprocess data
# 3. Train the model
# 4. Evaluate performance
```

## Methodology

### 1. Exploratory Data Analysis (EDA)

The project includes comprehensive visualizations:

- **Univariate Analysis:**
  - Distribution plots for Age, Salary, Experience, Performance Score
  - Count plots for categorical variables
  - Attrition rate analysis

- **Bivariate Analysis:**
  - Attrition by Department, Education, Gender
  - Attrition by Remote Work status
  - Salary and Experience distributions by Attrition
  - Correlation heatmap

- **Advanced Visualizations:**
  - Pairplots of numerical features
  - Boxplots for outlier detection
  - Joinplots for relationship analysis

### 2. Data Preprocessing

**Missing Value Handling:**
- Numerical columns: Filled with median values
- Categorical columns: Filled with mode values

**Feature Engineering:**
- Derived tenure: `2024 - Joining_Year`
- One-hot encoding for categorical variables (Gender, Education, Department, Remote_Work)
- Binary encoding for target variable (Attrition: Yes=1, No=0)

**Feature Scaling:**
- StandardScaler applied to numerical features (Age, Salary, Experience, Performance_Score, Certifications, Tenure)

**Train-Test Split:**
- 80% training data
- 20% test data
- Random state: 42

### 3. Model Implementation

**Logistic Regression from Scratch:**

```python
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross-entropy loss
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-9))

# Gradient descent
for iteration in range(iterations):
    z = X_train @ theta
    h = sigmoid(z)
    gradient = (X_train.T @ (h - y_train)) / m
    theta -= learning_rate * gradient
```

**Hyperparameters:**
- Learning rate: 0.01
- Iterations: 1000
- Regularization: None (can be added)

### 4. Model Evaluation

Multiple metrics are used to evaluate model performance:

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification results

## Results

Based on the notebook output:

```
Final Model Evaluation
------------------------
Test Loss: 0.6953
Accuracy: 0.4972 (49.72%)
Precision: 0.4839 (48.39%)
Recall: 0.5233 (52.33%)
F1 Score: 0.5028 (50.28%)
ROC-AUC: 0.5017

Confusion Matrix:
[[43 48]
 [41 45]]
```

### Model Interpretation

The model shows baseline performance (~50% accuracy), indicating that the attrition prediction task is challenging with the current features. This suggests:

1. **Balanced predictions**: The model doesn't heavily favor one class
2. **Feature limitations**: Additional features might be needed (job satisfaction, manager quality, commute time, etc.)
3. **Complex relationships**: Employee attrition may involve non-linear patterns that logistic regression cannot capture

### Key Insights from EDA

- **Salary Impact**: Lower salaries correlate with higher attrition
- **Experience**: Employees with less experience show higher attrition rates
- **Department Variations**: Attrition rates vary significantly across departments
- **Performance**: Interestingly, performance scores show varied correlation with attrition
- **Remote Work**: Remote work options impact retention differently across employee segments

## Features

✅ **Implemented:**
- Complete EDA with 15+ visualizations
- Custom logistic regression implementation
- Data preprocessing pipeline
- Missing value imputation
- Feature scaling and encoding
- Multiple evaluation metrics
- Training loss visualization
- Confusion matrix analysis

🔄 **Potential Enhancements:**
- Regularization (L1/L2)
- Feature selection algorithms
- Cross-validation
- Hyperparameter tuning
- Advanced models (Random Forest, XGBoost)
- SMOTE for class imbalance
- Feature importance analysis
- Interactive dashboards

## Dependencies

```txt
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Data Preprocessing Details

### Numerical Features Processing:
1. Missing values filled with median
2. Standardization using StandardScaler
3. Features: Age, Salary, Experience, Performance_Score, Certifications, Tenure

### Categorical Features Processing:
1. Missing values filled with mode
2. One-hot encoding applied
3. Original columns dropped after encoding
4. Features: Gender, Education, Department, Remote_Work

### Target Variable:
- Binary encoding: Yes → 1, No → 0
- No class imbalance handling (can be added)

## Model Training Process

1. **Initialization**: Weights initialized to zeros
2. **Forward Pass**: Calculate predictions using sigmoid function
3. **Loss Calculation**: Cross-entropy loss computed
4. **Backward Pass**: Gradient calculated
5. **Weight Update**: Gradient descent step
6. **Iteration**: Repeat for 1000 epochs
7. **Convergence**: Monitor training loss curve

## Evaluation Metrics Explained

- **Accuracy**: (TP + TN) / Total - Overall correctness
- **Precision**: TP / (TP + FP) - Quality of positive predictions
- **Recall**: TP / (TP + FN) - Coverage of actual positives
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall) - Balanced metric
- **ROC-AUC**: Model's ability to distinguish between classes

Where:
- TP = True Positives
- TN = True Negatives
- FP = False Positives
- FN = False Negatives

## Future Improvements

### Model Enhancements:
1. **Regularization**: Add L1/L2 penalty to prevent overfitting
2. **Advanced Models**: Implement Random Forest, Gradient Boosting, Neural Networks
3. **Ensemble Methods**: Combine multiple models
4. **Feature Engineering**: Create interaction terms, polynomial features

### Data Enhancements:
1. **More Features**: Collect job satisfaction, manager ratings, work-life balance scores
2. **Temporal Analysis**: Analyze attrition trends over time
3. **External Data**: Economic indicators, industry benchmarks

### Technical Improvements:
1. **Cross-Validation**: K-fold CV for robust evaluation
2. **Hyperparameter Tuning**: Grid search or random search
3. **Class Imbalance**: SMOTE, class weights, or downsampling
4. **Pipeline**: Create sklearn pipeline for reproducibility

### Visualization:
1. **Interactive Dashboards**: Plotly or Dash for interactive EDA
2. **SHAP Values**: Model interpretability
3. **ROC Curves**: Visual comparison of models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset generated for educational purposes
- Inspired by HR analytics and employee retention research
- Built with standard data science libraries (pandas, numpy, matplotlib, seaborn)

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is an educational project demonstrating logistic regression implementation from scratch. For production use, consider using optimized libraries like scikit-learn with additional feature engineering and model validation.
