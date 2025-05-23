import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap
import hdbscan

# Загружаем переменные окружения из .env файла
load_dotenv()

# Получаем путь к файлу относительно корня проекта
project_root = Path(__file__).parent.parent
data_file_path = project_root / "data" / "DECENTRATHON_3.0.parquet"
df = pd.read_parquet(data_file_path)
print("Данные загружены:", df.shape)

# === Пример агрегации по картам (пользователям) ===
df_users = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'count'],
    'merchant_city': pd.Series.nunique,
    'merchant_mcc': pd.Series.nunique,
    'wallet_type': pd.Series.nunique,
}).reset_index()

df_users.columns = [
    'card_id',
    'total_amount', 'avg_amount', 'transaction_count',
    'unique_cities', 'unique_mcc', 'unique_wallets'
]

# === Стандартизация ===
features = ['total_amount', 'avg_amount', 'transaction_count',
            'unique_cities', 'unique_mcc', 'unique_wallets']
X = df_users[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Кластеризация (KMeans) ===
kmeans = KMeans(n_clusters=4, random_state=42)
df_users['cluster'] = kmeans.fit_predict(X_scaled)

# === Визуализация (UMAP) ===
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_scaled)
df_users['x'] = embedding[:, 0]
df_users['y'] = embedding[:, 1]

# === График кластеров ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_users, x='x', y='y',
                hue='cluster', palette='Set2', s=70)
plt.title("Кластеры клиентов (card_id) — UMAP + KMeans")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(title="Кластер")
plt.grid(True)
plt.tight_layout()
plt.show()
