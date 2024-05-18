import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Loading data 
df =  pd.read_csv("/Users/kumargaurav/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()
# Data shape
df.shape
# Data types
df.dtypes
# Missing values
df.isna().sum()
# Data basic stats
df.describe()
# Drop unwanted columns
df.drop('customerID',axis='columns',inplace=True) 
#Class distribution
print(df.Churn.value_counts())
fig = px.pie(df, names=df["Churn"].map({"No":"Non-churn","Yes":"Churn"}), title='Population of Churn and Non-churn group')
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# gender distribution

print(df.gender.value_counts())
fig = px.pie(df, names=df["gender"])
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# gender distribution by class
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title('gender by Churn')
plt.xlabel('gender')
plt.ylabel('Churn')
plt.show()


# SeniorCitizen distribution
print(df.SeniorCitizen.value_counts())
fig = px.pie(df, names=df["gender"])
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# SeniorCitizen distribution by class
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='SeniorCitizen', hue='Churn', data=df)
plt.title('SeniorCitizen by Churn')
plt.xlabel('SeniorCitizen')
plt.ylabel('Churn')
plt.show()


# Partner distribution

print(df.SeniorCitizen.value_counts())
fig = px.pie(df, names=df["Partner"])
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# Partner distribution by class
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='Partner', hue='Churn', data=df)
plt.title('Partner by Churn')
plt.xlabel('Partner')
plt.ylabel('Churn')
plt.show()


# Dependents distribution

print(df.Dependents.value_counts())
fig = px.pie(df, names=df["Dependents"])
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# Dependents distribution by class
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='Dependents', hue='Churn', data=df)
plt.title('Dependents by Churn')
plt.xlabel('Dependents')
plt.ylabel('Churn')
plt.show()



# Contract distribution

print(df.Dependents.value_counts())
fig = px.pie(df, names=df["Contract"])
fig.update_traces(textinfo='value+percent', textfont_size=18)
fig.update_layout(width=700, height=500)
fig.show()

# Contract distribution by class
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Contract by Churn')
plt.xlabel('Contract')
plt.ylabel('Churn')
plt.show()


# Converting TotalCharges to numerical
df =  df[df["TotalCharges"] != ' ']
df["TotalCharges"] =  df["TotalCharges"].apply(float)


# tenure analysis for churn class
tenure_churn_no = df[df.Churn=='No'].tenure
tenure_churn_yes = df[df.Churn=='Yes'].tenure
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer churn prediction ")
plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.9, color=['blue','red'],label=['Churn=Yes','Churn=No'])
plt.legend()

# monthly charges analysis

monthly_charges_no = df[df.Churn=='No'].MonthlyCharges      
monthly_charges_yes = df[df.Churn=='Yes'].MonthlyCharges      
plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([monthly_charges_yes, monthly_charges_no], rwidth=0.8, color=['blue','red'],label=['Churn=Yes','Churn=No'])
plt.legend()


# Contract vs PaymentMethod analysis
cross_tab = pd.crosstab(df['Contract'], df['PaymentMethod'])
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='coolwarm')
plt.title('Heatmap of Contract vs. PaymentMethod')
plt.show()


# contact vs InternetService 
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='InternetService', hue='Contract', data=df)
plt.title('Contract by internet service Type')
plt.xlabel('Internet service Type')
plt.ylabel('Contract Tpe')
plt.show()



# PaymentMethod vs InternetService
sns.set(style="whitegrid")
plt.figure(figsize=(8,4))
sns.countplot(x='PaymentMethod', hue='InternetService', data=df)
plt.title('Internet service by payment method')
plt.xlabel('Payment Method')
plt.ylabel('Internet Service')
plt.show()



# function for printing unique values in each columns
def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column}: {df[column].unique()}') 


# unique values in each columns
print_unique_col_values(df)

# Replace No internet service and No phone service with no
df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)

print_unique_col_values(df)

# Converting yes no columns with 0 and 1
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df[col].replace({'Yes': 1,'No': 0},inplace=True)

df.head()

# converting male female in 0 and 1
df['gender'].replace({'Female':1,'Male':0},inplace=True)


# One hot encoding for categorica columns
df_dummies = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod'])
df_dummies.columns

# Data normalization for numerical columns

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scaler = MinMaxScaler()
df_dummies[cols_to_scale] = scaler.fit_transform(df_dummies[cols_to_scale])


#Train Test split
X = df_dummies.drop('Churn',axis='columns')
y = df_dummies["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, stratify=y)

# class distribution in each class
y_train.value_counts()

# Logistic regression  model Training

lr = LogisticRegression()

lr.fit(X_train,y_train)

# prediction on test data
y_pred_lr = lr.predict(X_test)

# Calculate score
acc_lr = lr.score(X_test, y_pred_lr)
print("Accuracy_lr", acc_lr)
# Calculate precision and recall
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
print("Precision_lr:", precision_lr)
print("Recall_lr:", recall_lr)


# Fine Tune logistic regression model

log_reg = LogisticRegression()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear', 'saga'],
              'class_weight': ['balanced', None]}
grid_search_lr = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_params_lr = grid_search_lr.best_params_

best_log_reg = LogisticRegression(**best_params_lr)
best_log_reg.fit(X_train, y_train)

# Fine tuned logistic regression model evaluation 

y_pred_lr_best = best_log_reg.predict(X_test)

# Calculate score
acc_lr_best = best_log_reg.score(X_test, y_test)
print("Accuracy_lr_best", acc_lr_best)
# Calculate precision and recall
precision_lr_best = precision_score(y_test, y_pred_lr_best)
recall_lr_best = recall_score(y_test, y_pred_lr_best)
print("Precision_lr_best:", precision_lr_best)
print("Recall_lr_best:", recall_lr_best)



# Train Random forest model
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# prediction on test data
y_pred_rf = rf_classifier.predict(X_test)

# Calculate score
acc_rf = rf_classifier.score(X_test, y_test)
print("Accuracy_fr", acc_rf)
# Calculate precision and recall
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
print("Precision_rf:", precision_rf)
print("Recall_rf:", recall_rf)


# Fine Tune the random forest model

rf_classifier1 = RandomForestClassifier()
param_grid_rf = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10]}
grid_search_rf = GridSearchCV(rf_classifier1, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
best_rf_classifier = RandomForestClassifier(**best_params_rf)
best_rf_classifier.fit(X_train, y_train)

# prediction on test data
y_pred_rf_best = best_rf_classifier.predict(X_test)

# Calculate score
acc_rf_best = best_rf_classifier.score(X_test, y_test)
print("Accuracy_rf_best", acc_rf_best)
# Calculate precision and recall
precision_rf_best = precision_score(y_test, y_pred_rf_best)
recall_rf_best = recall_score(y_test, y_pred_rf_best)
print("Precision_rf_best:", precision_rf_best)
print("Recall_rf_best:", recall_rf_best)



# Balancing the data with smote
smote = SMOTE(sampling_strategy='minority')
X_msote, y_smote = smote.fit_resample(X, y)

y_smote.value_counts()

# Train test split on balanced data
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_msote, y_smote, test_size=0.2, random_state=43, stratify=y_smote)


# Model training on balanced data
lr_model_smote = LogisticRegression()
lr_model_smote.fit(X_train_smote, y_train_smote)
# Model evaluation on smote data
y_pred_lr_smote = lr_model_smote.predict(X_test)

acc_lr_smote = lr_model_smote.score(X_test, y_test)
print("Accuracy_lr_smote", acc_lr_smote)
# Calculate precision and recall
precision_lr_smote = precision_score(y_test, y_pred_lr_smote)
recall_lr_smote = recall_score(y_test, y_pred_lr_smote)
print("Precision_lr_smote:", precision_lr_smote)
print("Recall_lr_smote:", recall_lr_smote)


# Fine tune logistic regression model on balanced data

log_reg_smote = LogisticRegression()

param_grid_lr_smote = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear', 'saga'],
              'class_weight': ['balanced', None]}
grid_search_lr_smote = GridSearchCV(log_reg_smote, param_grid_lr_smote, cv=5, scoring='accuracy')
grid_search_lr_smote.fit(X_train_smote, y_train_smote)
best_params_lr_smote = grid_search_lr_smote.best_params_

best_log_reg_smote = LogisticRegression(**best_params_lr_smote)
best_log_reg_smote.fit(X_train_smote, y_train_smote)

# Fine tuned logistic regression model evaluation 

y_pred_lr_best_smote = best_log_reg_smote.predict(X_test)

# Calculate score
acc_lr_best_smote = best_log_reg_smote.score(X_test, y_test)
print("Accuracy_lr_best_smote", acc_lr_best_smote)
# Calculate precision and recall
precision_lr_best_smote = precision_score(y_test, y_pred_lr_best_smote)
recall_lr_best_smote = recall_score(y_test, y_pred_lr_best_smote)
print("Precision_lr_best_smote:", precision_lr_best_smote)
print("Recall_lr_best_smote:", recall_lr_best_smote)





# Train Random forest model on balanced data
rf_classifier_smote = RandomForestClassifier()
rf_classifier_smote.fit(X_train_smote, y_train_smote)

# prediction on test data
y_pred_rf_smote = rf_classifier_smote.predict(X_test)

# Calculate score
acc_rf_smote= rf_classifier_smote.score(X_test, y_test)
print("Accuracy_rf_smote", acc_rf_smote)
# Calculate precision and recall
precision_rf_smote = precision_score(y_test, y_pred_rf_smote)
recall_rf_smote = recall_score(y_test, y_pred_rf_smote)
print("Precision_rf_smote:", precision_rf_smote)
print("Recall_rf_smote:", recall_rf_smote)



# Fine Tune the random forest model on balanced data

rf_classifier2 = RandomForestClassifier()
param_grid_rf_smote = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10]}
grid_search_rf_smote = GridSearchCV(rf_classifier2, param_grid_rf_smote, cv=5, scoring='accuracy')
grid_search_rf_smote.fit(X_train_smote, y_train_smote)
best_params_rf_smote =grid_search_rf_smote.best_params_
best_rf_classifier_smote = RandomForestClassifier(**best_params_rf_smote)
best_rf_classifier_smote.fit(X_train_smote, y_train_smote)

# prediction on test data
y_pred_rf_best_smote= best_rf_classifier_smote.predict(X_test)

# Calculate score
acc_rf_best_smote = best_rf_classifier_smote.score(X_test_smote, y_test_smote)
print("Accuracy_rf_best_smote", acc_rf_best_smote)
# Calculate precision and recall
precision_rf_best_smote = precision_score(y_test, y_pred_rf_best_smote)
recall_rf_best_smote = recall_score(y_test, y_pred_rf_best_smote)
print("Precision_rf_best_smote:", precision_rf_best_smote)
print("Recall_rf_best_smote:", recall_rf_best_smote)


























