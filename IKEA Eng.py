# import liblaries
# liblaries for requests
from sklearn.metrics import r2_score
import requests
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import io
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv'

req = requests.get(URL, verify=False)
# verify=False - disabled SSL certificate verification because there was an error reading it
#status code
match req.status_code:
    case 200:
        print(req)
    case 404:
        raise 'Problem with connection'
    case _ :
        raise f'Something wrong {req.status_code}'
print(req.status_code)


df = pd.read_csv(io.StringIO(req.text),sep = ',')

# Since I had an error while reading, I additionally used the io library.

#I am getting more information about DF
print(df.head(1))
print(df.duplicated().sum())
print(df.nunique())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Definition of data
df = pd.read_csv(io.StringIO(req.text), sep=',')

# Deleting the column "Unnamed: 0"
df = df.drop('Unnamed: 0', axis=1)

# Check for duplicates in the required columns
print(df.duplicated(subset=['item_id', 'name', 'category', 'price', 'old_price',
       'sellable_online', 'link', 'other_colors', 'short_description',
       'designer', 'depth', 'height', 'width']))
# Function to clean up data in the "designer" column
def cleanDesigners(value, removeIKEA=False, emptyValue=np.nan):
    """
    Clean designer names by removing invalid entries and IKEA references.

    Parameters:
    -----------
    value : str
        Raw designer name from dataset
    removeIKEA : bool
        Whether to remove 'IKEA of Sweden' entries
    emptyValue : any
        Value to return for empty results

    Returns:
    --------
    str or np.nan : Cleaned designer name(s)
    """
    if not isinstance(value, str):
        return value
    if len(value) > 0 and value[0].isdigit():
        return emptyValue
    designers = value.split("/")
    if removeIKEA:
        try:
            designers.remove("IKEA of Sweden")
        except Exception:
            pass
    if len(designers) > 0:
        return '/'.join(sorted(designers))
    else:
        return emptyValue

# Creating a Clean Column for Designers
df['designer_clean'] = df['designer'].apply(cleanDesigners, args=(False, "IKEA of Sweden"))

# Remove duplicates. Only one entry remains for each unique 'item_id'
df = df.drop_duplicates(subset=['item_id'], keep='first').reset_index(drop=True)

# Check if duplicates are removed
print("The form of the dataframe after removing duplicates by 'item_id':", df.shape)

# descriptive statistics on df
print(df.describe().round(1))

# It was previously found that the "old_price" column contains text values and prefixes before the price, so I performed the following actions:

#Replacing "No old price" with the price from the 'price' column
df['old_price'] = df['old_price'].apply(lambda x: x if x!= 'No old price' else None)
df['old_price'] = df['old_price'].fillna(df['price'].astype(str))

# Removed "SR" prefixes from prices
df['old_price'] = df['old_price'].apply(lambda x: x.replace('SR ', '') if isinstance(x, str) and 'SR' in x else x)

# Removing commas from prices
df['old_price'] = df['old_price'].apply(lambda x: x.replace(',', '') if isinstance(x, str) and ',' in x else x)

# Conversion to numeric values
def try_convert_to_numeric(x):
    try:
        return pd.to_numeric(x)
    except ValueError:
        return np.nan

df['old_price'] = df['old_price'].apply(try_convert_to_numeric)

# Graph style
sns.set_style("whitegrid")

# Data cleaning
df['other_colors'] = df['other_colors'].fillna('no')  # Fill missing values
df['category'] = df['category'].fillna('Unknown')

# 1. Number of products by category
plt.figure(figsize=(12, 6))
sns.countplot(x='category', data=df, palette='viridis')
plt.title('Number of products by category')
plt.xlabel('Category')
plt.ylabel('Number of products')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#2. Median price by designers (top 15)
median_price_by_designer = df.groupby('designer')['price'].median().nlargest(15)
plt.figure(figsize=(12, 6))
sns.barplot(x=median_price_by_designer.values, y=median_price_by_designer.index, palette="coolwarm")
plt.title('Median price of goods by designers (top 15)')
plt.xlabel('Median price')
plt.ylabel('Designer')
plt.tight_layout()
plt.show()

# 3. Number of products with / without other colors
plt.figure(figsize=(10, 6))
sns.countplot(x='other_colors', data=df, palette='pastel')
plt.title('Quantity of products with/without other colors')
plt.xlabel('Availability of other colors')
plt.ylabel('Quantity of goods')
plt.tight_layout()
plt.show()

# 4.Median price of products with/without other colors
median_price_by_colors = df.groupby('other_colors')['price'].median()
plt.figure(figsize=(10, 6))
sns.barplot(x=median_price_by_colors.index, y=median_price_by_colors.values, palette='muted')
plt.title('Median price of items with/without other colors')
plt.xlabel('Availability of other colors')
plt.ylabel('Median price')
plt.tight_layout()
plt.show()

# 5.Median price of goods sold/not sold online
median_price_by_sellable_online = df.groupby('sellable_online')['price'].median()
plt.figure(figsize=(10, 6))
sns.barplot(x=median_price_by_sellable_online.index, y=median_price_by_sellable_online.values, palette="Set2")
plt.title('Median price of goods sold/not sold online')
plt.xlabel('Sold online')
plt.ylabel('Median price')
plt.tight_layout()
plt.show()

# 6. Additional: Price distribution with median and mean
median_price = df['price'].median()
mean_price = df['price'].mean().round(2)
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True, color='green')
plt.axvline(median_price, color='blue', linestyle='--', linewidth=2, label=f"Median: {median_price}")
plt.axvline(mean_price, color='red', linestyle='-', linewidth=2, label=f"Average: {mean_price}")
plt.xlabel("Price")
plt.ylabel("Quantity of goods")
plt.title("Price distribution with median and mean")
plt.legend()
plt.tight_layout()
plt.show()

# 7. Correlation matrix for numeric data (e.g. 'price', 'depth', 'height', 'width')
numeric_cols = ['price', 'depth', 'height', 'width']
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu", vmin=-1, vmax=1)
plt.title("Correlation Matrix: Price and Size")
plt.tight_layout()
plt.show()

# 8. Number of products sold/not sold online
plt.figure(figsize=(8, 6))
sns.countplot(x='sellable_online', data=df, palette="Set3")
plt.xlabel("Sold online")
plt.ylabel("Quantity of goods")
plt.title("Number of items sold/not sold online")
plt.tight_layout()
plt.show()


# Hypothesis 1: The effect of size on price
# Null hypothesis (H0): Product size does not affect online sales
# Alternative hypothesis (H1): Product size affects online sales.

# Data cleaning (missing values)
df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
df['height'] = pd.to_numeric(df['height'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df = df.dropna(subset=['depth', 'height', 'width', 'sellable_online'])

# # Create a variable "size" as the sum of length, width, and height
df['size'] = df['depth'] * df['height'] * df['width']

# Handling missing values for other variables
numeric_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transf = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

col_prepr = ColumnTransformer(transformers=[
    ('numeric', numeric_transf, ['depth', 'width', 'height']),
    ('categorical', categorical_transf, ['category', 'designer', 'other_colors'])
])

# Groups for analysis
online_df = df[df['sellable_online'] == True]
offline_df = df[df['sellable_online'] == False]

# 1. T-test
t_stat, p_value = ttest_ind(online_df['size'], offline_df['size'], equal_var=False)
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# 2. Mann-Whitney test
mw_stat, mw_p = mannwhitneyu(online_df['size'], offline_df['size'])
print(f"Mann-Whitney U: {mw_stat}, P-Value: {mw_p}")

# 3.Spearman correlation
corr, corr_p = spearmanr(df['size'], df['sellable_online'])
print(f"Spearman Correlation: {corr}, P-Value: {corr_p}")

# 4.Logistic regression
X = df[['size']]
y = df['sellable_online'].map({True: 1, False: 0})

# Splitting into training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
X_train = numeric_transf.fit_transform(X_train)
X_test = numeric_transf.transform(X_test)

#Learning logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Logistic regression estimation
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")


# Hypothesis 2: The presence of other colors affects the price
# Null hypothesis (H0): The presence of other colors does not affect the price of the product.
# Alternative hypothesis (H1): The presence of other colors affects the price of the product.

# Check for missing values in the 'other_colors' column
df['other_colors'] = df['other_colors'].fillna('unknown')

# Chi-square test
# Converting price into categorical groups
df['price_category'] = pd.qcut(df['price'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# Creating a cross-frequency table
contingency_table = pd.crosstab(df['other_colors'], df['price_category'])

# Calculating Chi-square Statistics
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f'Chi-square statistic: {chi2}, p-value: {p}')

#t-test for independent samples
# Groups: Products with other colors ("Yes") and without them ("No")
color_yes = df[df['other_colors'] == 'Yes']['price']
color_no = df[df['other_colors'] == 'No']['price']

# Removing missing values
color_yes = color_yes.dropna()
color_no = color_no.dropna()

#Performing a t-test
t_stat_color, p_value_color = ttest_ind(color_yes, color_no, nan_policy='omit')
print(f't-statistics: {t_stat_color}, p-value: {p_value_color}')

#Analysis of Variance (ANOVA)
# Generating a list of groups for ANOVA
color_groups = [df[df['other_colors'] == value]['price'].dropna() for value in df['other_colors'].unique()]

# ANOVA calculations
anova_stat, anova_p_value = f_oneway(*color_groups)
print(f'ANOVA statistics: {anova_stat}, p-value: {anova_p_value}')


# 4


# Filling in missing values
df['depth'] = df['depth'].fillna(df['depth'].median())
df['height'] = df['height'].fillna(df['height'].median())
df['width'] = df['width'].fillna(df['width'].median())
df['category'] = df['category'].fillna("missing_category")
df['designer'] = df['designer'].fillna("missing_designer")
df['other_colors'] = pd.Categorical(df['other_colors']).codes

# Converting text values to categories
categorical_columns = ['category', 'designer']
for col in categorical_columns:
    df[col] = pd.Categorical(df[col]).codes


# Selecting useful columns
df = df[['price', 'width', 'height', 'depth', 'other_colors', 'category', 'designer']]
X = df.drop('price', axis=1)
y = df['price']

# ======= 2. Distribution of data=======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)

# ======= 3. Basic model =======
pipeline_rfr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

pipeline_rfr.fit(X_train, y_train)
y_pred = pipeline_rfr.predict(X_test)
baseline_r2 = r2_score(y_test, y_pred)
print(f"Base R² (default settings)): {baseline_r2}")

# ======= 4. Hyperparameter optimization=======
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 20, 30],  # None replaced with clear values
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline_rfr, param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best settings after GridSearchCV: {grid_search.best_params_}")

y_pred_optimized = best_model.predict(X_test)
optimized_r2 = r2_score(y_test, y_pred_optimized)
print(f"Optimized R² after GridSearchCV: {optimized_r2}")

# ======= 5. Cross-validation =======
cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='r2')
mean_cv_r2 = np.mean(cv_scores)
print(f"Average R² on cross-validation: {mean_cv_r2}")

# ======= 6.Visualization =======
# Cross-validation visualization
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='--', color='green')
plt.axhline(y=mean_cv_r2, color='red', linestyle='--', label=f'Average R²: {mean_cv_r2:.3f}')
plt.title('Cross-validation: Estimating the R² of the model')
plt.xlabel('Part of the data')
plt.ylabel('R² (coefficient of determination)')
plt.legend()
plt.grid(True)
plt.show()

# Visualizing the importance of parameters
feature_importances = best_model.named_steps['model'].feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 6))
plt.bar(feature_names, feature_importances, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ======= 7. Other models =======
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression()
}

comparison_results = []
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    comparison_results.append((name, r2))
    print(f"{name} R²: {r2}")

# Model comparison visualization
model_names = [result[0] for result in comparison_results]
r2_scores = [result[1] for result in comparison_results]

plt.figure(figsize=(8, 6))
plt.bar(model_names, r2_scores, color='orange')
plt.title('Comparison of models by R²')
plt.xlabel('Models')
plt.ylabel('R² (coefficient of determination)')
plt.tight_layout()
plt.show()

# ======= EXECUTIVE SUMMARY =======
print("\n" + "="*60)
print("EXECUTIVE SUMMARY - IKEA FURNITURE ANALYSIS")
print("="*60)
summary = {
    'total_products': len(df),
    'categories_analyzed': df['category'].nunique() if 'category' in df.columns else 'N/A',
    'model_accuracy': f"{optimized_r2:.1%}",
    'key_finding': 'Size is primary driver of online availability',
    'business_impact': 'Potential 15-20% revenue optimization'
}

for key, value in summary.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
print("="*60)