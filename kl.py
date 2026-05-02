import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = train[features].copy()
y = train['Survived']
X_test = test[features].copy()

X['Sex'] = (X['Sex'] == 'female').astype(int)
X_test['Sex'] = (X_test['Sex'] == 'female').astype(int)

X['Age'] = X['Age'].fillna(X['Age'].median())
X_test['Age'] = X_test['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())
X_test['Fare'] = X_test['Fare'].fillna(X['Fare'].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

print(f"Средняя точность: {cv_scores.mean():.4f}")

inertia = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

diff = np.diff(inertia)
optimal_k = K_range[np.argmin(diff) + 1]
print(f"\nОптимальное количество кластеров (метод локтя): {optimal_k}")

plt.figure(figsize=(6, 4))
plt.plot(K_range, inertia, 'bo-', linewidth=2)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'K = {optimal_k}')
plt.xlabel('Количество кластеров (K)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

for i in range(optimal_k):
    data = train[clusters == i]
    if len(data) > 0:
        print(f"\nКластер {i + 1}:")
        print(f"  Людей: {len(data)}")
        print(f"  Выжило: {data['Survived'].mean():.1%}")
        print(f"  Женщин: {(data['Sex'] == 'female').sum()} ({(data['Sex'] == 'female').mean():.1%})")
        print(f"  Класс билета: {data['Pclass'].mean():.1f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title(f'Кластеры KMeans (K={optimal_k})')

plt.subplot(1, 2, 2)
colors = ['red' if x == 0 else 'green' for x in y]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
plt.title('Истинные метки (красный=погиб, зеленый=выжил)')

plt.show()
