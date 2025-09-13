import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('extracted_lexical_features.csv')  # Output of LFTK Feature Extraction

df = df.drop(['X', 'topic_id_original_categorical','grade','learner_id','topic_id_categorical'],axis=1) #Drop unneccessary columns

#Run LASSO L1 Regulation

# Load the data
data = df

data = data.drop(['cefr_numeric'],axis=1)

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
alpha = 0.00077426
C_value = 1/alpha

# Logistic Regression with L1 regularization (LASSO) for multi-class classification
lasso_classifier_multi = LogisticRegression(C=C_value, penalty='l1', solver='saga', max_iter=5000, multi_class='multinomial', random_state=42)

# Creating and evaluating the pipeline for classification with multi-class target
pipeline_classification_multi = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', lasso_classifier_multi)
])

# Fit the LASSO classifier model for multi-class
pipeline_classification_multi.fit(X, y)

# Retrieve the coefficients from the model for each CEFR level
coefficients_multi = pipeline_classification_multi.named_steps['classifier'].coef_

# Pairing feature names with their coefficients for each CEFR level
feature_names = X.columns
feature_importance_multi = {}
cefr_levels = y.unique()
for index, level in enumerate(cefr_levels):
    feature_importance_multi[level] = dict(zip(feature_names, coefficients_multi[index]))

# Filtering out features with coefficients close to zero for each CEFR level
important_features_multi = {}
for level, importance in feature_importance_multi.items():
    important_features_multi[level] = {k: v for k, v in importance.items() if abs(v) > 1e-4}

print(important_features_multi)

lasso_results = pd.DataFrame(important_features_multi) #Save LASSO Results

lasso_results_descriptive_stats = lasso_results.describe()

lasso_results_descriptive_stats

	A1	A2	B1	B2	C1
count	76.000000	76.000000	76.000000	76.000000	76.000000
mean	-0.055795	-0.044757	0.011938	0.045372	0.043235
std	0.645709	0.879296	0.290587	1.052332	0.501384
min	-3.665669	-5.538792	-0.636739	-5.085169	-0.771400
25%	-0.143822	-0.158646	-0.136914	-0.100181	-0.191640
50%	-0.008720	0.014013	0.019405	-0.012828	-0.062880
75%	0.157528	0.179899	0.118530	0.093528	0.160022
max	1.962092	4.065541	1.147783	6.193364	2.476290

df_filtered = lasso_results [~lasso_results ['Unnamed: 0'].str.startswith('l1')] #Filter lasso results to drop row corresponding learners' L1. We dropped them due to insignificant values.

# Sort each CEFR level column in descending order to identify the top 10 lexical complexity measures
top_10_measures = {}
cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1']

for level in cefr_levels:
    top_10_measures[level] = df_filtered.nlargest(10, level)[['Unnamed: 0', level]]

top_10_measures

{'A1':            Unnamed: 0        A1
 2   t_subtlex_us_zipf  1.962092
 60          bilog_ttr  0.747585
 65   bilog_ttr_no_lem  0.747585
 33       root_num_var  0.613264
 57           simp_ttr  0.592400
 62    simp_ttr_no_lem  0.592400
 32      root_noun_var  0.569833
 14       simp_det_var  0.534422
 48      corr_noun_var  0.528437
 29     root_cconj_var  0.418608,
 'A2':               Unnamed: 0        A2
 1                  t_bry  4.065541
 48         corr_noun_var  0.723360
 32         root_noun_var  0.653732
 45        corr_cconj_var  0.589887
 56         corr_verb_var  0.534011
 3               a_kup_pw  0.408652
 10          simp_adp_var  0.381529
 27          root_adv_var  0.335485
 7   a_subtlex_us_zipf_pw  0.290783
 34         root_part_var  0.272196,
 'B1':               Unnamed: 0        B1
 60             bilog_ttr  1.147783
 65      bilog_ttr_no_lem  1.147783
 0                  t_kup  0.534405
 3               a_kup_pw  0.407939
 1                  t_bry  0.397255
 39          root_sym_var  0.290714
 30          root_det_var  0.286319
 46          corr_det_var  0.251716
 40         root_verb_var  0.235495
 8   a_subtlex_us_zipf_ps  0.231225,
 'B2':          Unnamed: 0        B2
 0             t_kup  6.193364
 4          a_bry_pw  2.662628
 59         corr_ttr  0.920008
 64  corr_ttr_no_lem  0.920008
 58         root_ttr  0.915552
 63  root_ttr_no_lem  0.915552
 24    simp_verb_var  0.615490
 57         simp_ttr  0.559356
 62  simp_ttr_no_lem  0.559356
 13   simp_cconj_var  0.340827,
 'C1':               Unnamed: 0        C1
 0                  t_kup  2.476290
 3               a_kup_pw  1.038532
 59              corr_ttr  0.990576
 64       corr_ttr_no_lem  0.990576
 58              root_ttr  0.980956
 63       root_ttr_no_lem  0.980956
 7   a_subtlex_us_zipf_pw  0.862225
 1                  t_bry  0.787292
 11          simp_adv_var  0.572737
 24         simp_verb_var  0.559374}


combined_df = pd.concat(top_10_measures, keys=cefr_levels).reset_index().rename(columns={'level_0': 'CEFR Level', 'level_1': 'Rank'}) #Combine Results with CEFR level per Top Ten Features.
combined_df['Rank'] += 1  # Adjust the rank to start from 1 instead of 0
#standardize LASSO Results. There are two alternatives here. You can either run over all LASSO results rather than focus on Top Ten or only focus on Top Ten Features.

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Standardize the scores of the top 10 measures per CEFR level
standardized_dfs = {}

for level in cefr_levels:
    # Extract the measures and their L1 coefficients
    measures_df = top_10_measures[level].copy()
    
    # Standardize the L1 coefficients
    measures_df[level] = scaler.fit_transform(measures_df[[level]])
    
    standardized_dfs[level] = measures_df

# Combine the standardized measures into a single DataFrame for easier interpretation
combined_standardized_df = pd.concat(standardized_dfs, keys=cefr_levels).reset_index().rename(columns={'level_0': 'CEFR Level', 'level_1': 'Rank'})
combined_standardized_df['Rank'] += 1  # Adjust the rank to start from 1 instead of 0

combined_standardized_df.head(30)

#Plot Results (Bar Plots)

import matplotlib.pyplot as plt
import seaborn as sns

# Standardize the LASSO L1 coefficients
df_standardized = combined_standardized_df.copy()
cefr_levels = ['A1', 'A2', 'B1', 'B2', 'C1']


# Sort by absolute value of standardized coefficients and select top 10 indices for each CEFR level
top_indices = {}
for level in cefr_levels:
    top_indices[level] = df_standardized[[level, 'Unnamed: 0']].dropna().sort_values(by=level, key=lambda x: abs(x), ascending=False).head(10)

# Create subplots for each CEFR level, aligned from left to right, with measures listed vertically on the y-axis
fig, axes = plt.subplots(1, len(cefr_levels), figsize=(20, 10))
fig.suptitle('Top 10 Standardized LASSO L1 Coefficients for Lexical Complexity Indices per CEFR Level')

for i, level in enumerate(cefr_levels):
    ax = axes[i]
    sns.barplot(x=level, y='Unnamed: 0', data=top_indices[level], ax=ax, palette='blend:#7AB,#EDA')
    ax.set_title(f'CEFR Level {level}')
    
    # Remove x-axis labels for clarity
    ax.set_xlabel('')
    
    # Label only the y-axis with measures
    ax.set_yticklabels(top_indices[level]['Unnamed: 0'])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Create a DataFrame suitable for the heatmap using the standardized L1 coefficients of the top 10 measures for each CEFR level
heatmap_df = pd.concat([top_indices[level].set_index('Unnamed: 0')[[level]] for level in cefr_levels], axis=1)

# Create the heatmap using the standardized L1 coefficients
plt.figure(figsize=(14, 12))
sns.heatmap(heatmap_df, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True), cbar_kws={'label': 'Standardized L1 Coefficient'})
plt.title('Top 10 Standardized LASSO L1 Coefficients for Lexical Complexity Measures per CEFR Level')
plt.xlabel('CEFR Level')
plt.ylabel('Lexical Complexity Measure')
plt.tight_layout()
plt.savefig('lasso_l1_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Save all results
lasso_results.to_csv('lasso_results.csv', index=False)
combined_standardized_df.to_csv('standardized_lasso_results.csv', index=False)
print("LASSO L1 analysis completed and results saved!")


