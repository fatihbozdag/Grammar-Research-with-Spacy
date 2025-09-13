from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import KNNImputer

df = pd.read_csv('extracted_lexical_features.csv')  # Import LASSO Results
combined_std_lasso_results = pd.read_csv('standardized_lasso_results.csv')  # Import Standardized Top Ten LASSO Scores

#Filter DataFrame to include Top Ten features per CEFR

selected_features = combined_std_lasso_results['Unnamed: 0'].unique()
existing_features = [feature for feature in selected_features if feature in df.columns]
filtered_df = df[['cefr_numeric'] + existing_features]

#Since Random Forest is not happy with NaN values, you need to impute

from sklearn.impute import KNNImputer
# Imputing NaN values using KNN
knn_imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors
data_imputed = pd.DataFrame(knn_imputer.fit_transform(filtered_df), columns=filtered_df.columns)

#Run Random Forest and Standarlize Results
# Features (X) and target variable (y)
X = data_imputed.drop(columns=['cefr_numeric'])
y = data_imputed['cefr_numeric']

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Random Forest classifier with a reduced number of estimators (50)
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predicting the target variable on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred)
classification_rep_rf = classification_report(y_test, y_pred, target_names=['A1', 'A2', 'B1', 'B2', 'C1'])

# Print the accuracy and classification report
print("Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_rep_rf)

# Extract the feature importances
feature_importances = rf_classifier.feature_importances_
features = X.columns

# Combine the features and their importances into a DataFrame
feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Normalize the feature importances
scaler = MinMaxScaler()
feature_importances_df['NormalizedImportance'] = scaler.fit_transform(feature_importances_df[['Importance']])

# Sort the DataFrame to get the top 10 most important features
top_10_features = feature_importances_df.nlargest(10, 'NormalizedImportance')

# Visualize the top 10 feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['NormalizedImportance'], color='lightsteelblue')
plt.xlabel('Normalized Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Top 10 Normalized Feature Importances')
plt.gca().invert_yaxis()  # Invert the y-axis to have the most important features on top
plt.show()

# Initialize an empty DataFrame to store the feature importances for each CEFR level
all_features_df = pd.DataFrame(columns=['CEFR_Level', 'Feature', 'Importance'])

# Loop through each unique CEFR level
for cefr_level in sorted(y.unique()):
    # Create binary target variable for the specific CEFR level
    y_binary = (y == cefr_level).astype(int)

    # Split data into training and testing sets
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Create and train Random Forest classifier
    rf_classifier_binary = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_classifier_binary.fit(X_train, y_train_binary)

    # Extract and normalize feature importances
    feature_importances_binary = rf_classifier_binary.feature_importances_
    normalized_importances = feature_importances_binary / feature_importances_binary.sum()

    # Create DataFrame for features for this CEFR level
    cefr_features = pd.DataFrame({
        'CEFR_Level': [cefr_level] * len(X.columns),
        'Feature': X.columns,
        'Importance': normalized_importances
    })

    # Append to main DataFrame
    all_features_df = pd.concat([all_features_df, cefr_features], ignore_index=True)


# Create a mapping from numerical CEFR levels to string representation
cefr_mapping = {
    1.0: 'A1',
    2.0: 'A2',
    3.0: 'B1',
    4.0: 'B2',
    5.0: 'C1'
}

# Generate the plot
fig, axes = plt.subplots(1, len(all_features_df['CEFR_Level'].unique()), figsize=(20, 10))
fig.suptitle('Top 10 Standardized Random Forest Results for Lexical Complexity Indices per CEFR Level', fontsize=16)

# Loop through each CEFR level and plot the standardized importance of the top 10 features
for i, cefr_level in enumerate(sorted(all_features_df['CEFR_Level'].unique())):
    cefr_data = all_features_df[all_features_df['CEFR_Level'] == cefr_level]
    cefr_data = cefr_data.nlargest(10, 'Importance').sort_values(by='Importance', ascending=False)
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=cefr_data, 
        ax=axes[i],
        palette="blend:#7AB,#EDA")
    
    axes[i].set_title(f'CEFR Level {cefr_mapping[cefr_level]}')  # Use the mapping to set the title
    axes[i].grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    axes[i].set_xlabel('')
    axes[i].set_yticklabels(cefr_data['Feature'])  # Set y-tick labels for each subplot

# Show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Reimport LASSO Results for Random Forest for L1
# Load your DataFrame with raw scores
raw_scores_df = df
#filter_df
selected_features = combined_std_lasso_results['Unnamed: 0'].unique()
existing_features = [feature for feature in selected_features if feature in df.columns]
filtered_df = df[['cefr_numeric', 'l1','cefr'] + existing_features]

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Initialize a dictionary to store feature importances for each first language
feature_importances_per_l1 = {}

# Initialize the imputer
imputer = SimpleImputer(strategy='mean')

# Loop through each first language to fit a Random Forest model
for l1 in filtered_df['l1'].unique():
    l1_data = filtered_df[filtered_df['l1'] == l1]
    
    # Impute missing values
    l1_data_imputed = pd.DataFrame(imputer.fit_transform(l1_data.drop(['l1', 'cefr', 'cefr_numeric'], axis=1)), columns=l1_data.drop(['l1', 'cefr', 'cefr_numeric'], axis=1).columns)
    
    X = l1_data_imputed
    y = l1_data['cefr_numeric'].reset_index(drop=True)
    
    if len(X) < 10:
        continue
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    feature_importances_per_l1[l1] = feature_importances

# Normalize the feature importances
scaler = MinMaxScaler()
all_importances = pd.concat(feature_importances_per_l1.values())['Importance']
normalized_importances = scaler.fit_transform(all_importances.values.reshape(-1, 1))

i = 0
for l1, df in feature_importances_per_l1.items():
    n = len(df)
    df['NormalizedImportance'] = normalized_importances[i:i+n]
    i += n

# Consolidated plot
consolidated_df = pd.concat([df.assign(L1=l1) for l1, df in feature_importances_per_l1.items()])
plt.figure(figsize=(15, 15))
sns.barplot(data=consolidated_df, x='NormalizedImportance', y='Feature', hue='L1', palette='blend:#7AB,#EDA')
plt.title('Standardized Random Forest Results for Lexical Complexity Indices Across First Languages')
plt.tight_layout()
plt.savefig('random_forest_l1_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save results
consolidated_df.to_csv('random_forest_l1_results.csv', index=False)
print("Random Forest analysis completed and results saved!")
