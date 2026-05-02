import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("disease_train.csv")
test = pd.read_csv("disease_public_test.csv")
sample = pd.read_csv("disease_sample_submission.csv")

X = train.drop("Y", axis=1)
y = train["Y"]
X = X.fillna(X.median())
test = test.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=5000, learning_rate='adaptive', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
val_acc = accuracy_score(y_val, model.predict(X_val))
cv_mean = cross_val_score(model, X_scaled, y, cv=5).mean()
test_acc = accuracy_score(sample["Y"], model.predict(test_scaled))


print(f"Точность на валидации: {val_acc:.4f}")
print(f"Кросс-валидация: {cv_mean:.4f}")
print(f"Точность на тестовом наборе данных: {test_acc:.4f}")
