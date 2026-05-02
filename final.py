import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")

X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = train["Survived"]
X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
X_test['Sex'] = LabelEncoder().fit(train['Sex']).transform(X_test['Sex'])

X['Age'] = X['Age'].fillna(X['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X_test['Fare'] = X_test['Fare'].fillna(X['Fare'].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=5000, learning_rate='adaptive', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

val_acc = accuracy_score(y_val, model.predict(X_val))
cv_mean = cross_val_score(model, X_scaled, y, cv=5).mean()
test_acc = accuracy_score(gender["Survived"], model.predict(X_test_scaled))

print(f"Точность на валидации: {val_acc:.4f}")
print(f"Кросс-валидация: {cv_mean:.4f}")
print(f"Точность на тестовом наборе данных: {test_acc:.4f}")
