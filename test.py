import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("disease_train.csv")
test = pd.read_csv("disease_public_test.csv")
sample = pd.read_csv("disease_sample_submission.csv")

print("РАЗМЕРЫ ДАТАСЕТОВ")
print("train:", train.shape)
print("test:", test.shape)
print("sample:", sample.shape)

print("\nКОЛОНКИ")
print("train:", list(train.columns))
print("test:", list(test.columns))
print("sample:", list(sample.columns))

X = train.drop("Y", axis=1)
y = train["Y"]

print("\nПРОПУСКИ")
print(X.isna().sum())

median = X.median()
X = X.fillna(median)
test = test.fillna(median)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test)

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
    max_iter=20000,
    tol=1e-4,
    early_stopping=False,
    random_state=42
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)

print("\nРЕЗУЛЬТАТЫ НА VALIDATION")
print("Accuracy:", accuracy_score(y_val, val_pred))
print("Матрица ошибок:\n", confusion_matrix(y_val, val_pred))

test_pred = model.predict(X_test)

print("\nПРОВЕРКА TEST vs SAMPLE")

print("Длина test_pred:", len(test_pred))
print("Длина sample Y:", len(sample))

try:
    acc = accuracy_score(sample["Y"], test_pred)
    print("Accuracy на sample:", acc)
except Exception as e:
    print("Ошибка сравнения:", e)

print("\nСРЕДНИЕ ЗНАЧЕНИЯ (проверка сдвига данных)")
print("Train mean:\n", X.mean().values)
print("Test mean:\n", test.mean().values)

print("\nПРОВЕРКА СООТВЕТСТВИЯ СТРОК")
print("Индексы совпадают:", (test.index == sample.index).all())