import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

df = pd.read_csv('dataset01234.csv').drop('Unnamed: 0',axis=1)
df = df.sample(frac=1).reset_index(drop=True)
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)
rfModel = RandomForestClassifier()
rfModel.fit(X_train, y_train)
y_pred = rfModel.predict(X_test)
score = accuracy_score(y_pred=y_pred,y_true=y_test)
print(f"Accuracy Score : {score*100}%")
import joblib

# save
joblib.dump(rfModel, "rfModel01234.joblib")