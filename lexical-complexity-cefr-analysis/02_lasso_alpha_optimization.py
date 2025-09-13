import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

df = pd.read_csv('extracted_lexical_features.csv')  # Import raw lexical complexity results

df = df.drop(['X', 'topic_id_original_categorical', 'cefr_numeric','grade','learner_id', 'topic_id_categorical'],axis=1)


# Load the data
data = df

# Dummy code the categorical columns
data_dummies = pd.get_dummies(data, columns=['l1'], drop_first=True)

# Features and target
X = data_dummies.drop(columns=['cefr'])  # Excluding only the target column
y = data['cefr']

# Preprocessing: Scaling all predictor columns
preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), X.columns)
    ],
    remainder='passthrough'
)

# Logistic Regression with L1 regularization (LASSO) for multi-class classification using cross-validation
lasso_classifier_multi_cv = LogisticRegressionCV(
    penalty='l1', 
    solver='saga', 
    max_iter=5000, 
    multi_class='multinomial', 
    random_state=42, 
    cv=5  # Using 5-fold cross-validation; you can adjust this
)

# Creating and evaluating the pipeline for classification with multi-class target
pipeline_classification_multi_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lasso_classifier_multi_cv)
])

# Fit the model
pipeline_classification_multi_cv.fit(X, y)

# Extract the best C value
best_C = lasso_classifier_multi_cv.C_

print("Best C values for each class:", best_C)

# Calculate corresponding alpha values
alpha_values = 1 / best_C
print("Corresponding alpha values:", alpha_values)

# Save the results
results = pd.DataFrame({
    'class': y.unique(),
    'best_C': best_C,
    'alpha_values': alpha_values
})

results.to_csv('lasso_alpha_optimization_results.csv', index=False)
print("Alpha optimization results saved to 'lasso_alpha_optimization_results.csv'")
