import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

train = pd.read_csv("disease_train.csv")
test = pd.read_csv("disease_public_test.csv")
sample = pd.read_csv("disease_sample_submission.csv")

X = train.drop("Y", axis=1)
y = train["Y"]

X = X.fillna(X.median())
test = test.fillna(X.median())

X = np.log1p(X)
test = np.log1p(test)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test = scaler.transform(test)

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,
    max_iter=20000,
    tol=1e-4,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, pred))
print(confusion_matrix(y_val, pred))

test_pred = model.predict(test)
print("Test accuracy:", accuracy_score(sample["Y"], test_pred))

