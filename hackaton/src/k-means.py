# Облегченный анализ банковских транзакций для больших данных
# Автор: Erik (Decentra) - Оптимизированная версия для стабильной работы

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# Настройка для экономии памяти
pd.set_option('display.precision', 2)
plt.style.use('default')  # Простой стиль

print("🚀 ОБЛЕГЧЕННЫЙ АНАЛИЗ БАНКОВСКИХ ТРАНЗАКЦИЙ")
print("=" * 55)
print("🎯 Цель: Стабильная кластеризация больших данных")
print("⚡ Подход: Умная выборка + оптимизированные алгоритмы")

# 1. ЗАГРУЗКА ДАННЫХ
print("\n📊 Шаг 1: Загрузка данных...")

data_path = "/kaggle/input/decentra"
parquet_files = []

if os.path.exists(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.parquet'):
            parquet_files.append(os.path.join(data_path, filename))
            print(f"  📁 Найден: {filename}")

df = None
if parquet_files:
    file_path = parquet_files[0]
    print(f"Загружаем: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Данные загружены: {df.shape[0]:,} записей, {df.shape[1]} столбцов")
        print(f"📊 Уникальных клиентов: {df['card_id'].nunique():,}")
        
        # Проверяем размер в памяти
        memory_usage = df.memory_usage(deep=True).sum() / 1024**3
        print(f"💾 Размер в памяти: {memory_usage:.1f} GB")
        
        if memory_usage > 8:
            print("⚠️ Данные очень большие - будем использовать выборку")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        df = None

if df is None:
    print("🚫 Данные не загружены, завершаем анализ")
    exit()

# 2. БЫСТРАЯ ОЧИСТКА ДАННЫХ
print("\n🧹 Шаг 2: Быстрая очистка данных...")

original_size = len(df)
df = df[df['transaction_amount_kzt'] > 0]
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')
df = df[df['transaction_timestamp'].notna()]

print(f"Очистка: {original_size:,} → {len(df):,} записей")

# Добавляем только критически важные временные признаки
df['hour'] = df['transaction_timestamp'].dt.hour
df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 3. УМНАЯ ВЫБОРКА КЛИЕНТОВ
print("\n🎯 Шаг 3: Создание стратифицированной выборки...")

# Определяем максимальное количество клиентов для анализа
MAX_CLIENTS = 25000
unique_clients_total = df['card_id'].nunique()

print(f"Всего уникальных клиентов: {unique_clients_total:,}")

if unique_clients_total > MAX_CLIENTS:
    print(f"⚠️ Слишком много клиентов, создаем выборку из {MAX_CLIENTS:,}")
    
    # Быстрая стратификация по активности
    client_activity = df.groupby('card_id').agg({
        'transaction_amount_kzt': ['count', 'sum'],
        'transaction_timestamp': ['min', 'max']
    })
    
    client_activity.columns = ['txn_count', 'total_amount', 'first_txn', 'last_txn']
    client_activity['activity_days'] = (client_activity['last_txn'] - client_activity['first_txn']).dt.days + 1
    
    # Создаем категории активности
    client_activity['activity_level'] = pd.cut(
        client_activity['txn_count'], 
        bins=[0, 10, 50, 200, float('inf')], 
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    
    # Пропорциональная выборка
    sample_clients = []
    for level in ['Low', 'Medium', 'High', 'VeryHigh']:
        level_clients = client_activity[client_activity['activity_level'] == level].index
        sample_size = min(MAX_CLIENTS // 4, len(level_clients))
        if sample_size > 0:
            sample = np.random.choice(level_clients, sample_size, replace=False)
            sample_clients.extend(sample)
    
    # Дополняем до MAX_CLIENTS случайными клиентами
    remaining = MAX_CLIENTS - len(sample_clients)
    if remaining > 0:
        all_other = client_activity.index.difference(sample_clients)
        if len(all_other) > 0:
            additional = np.random.choice(all_other, min(remaining, len(all_other)), replace=False)
            sample_clients.extend(additional)
    
    # Фильтруем исходные данные
    df = df[df['card_id'].isin(sample_clients)]
    print(f"✅ Выборка: {len(sample_clients):,} клиентов, {len(df):,} транзакций")
    
    # Освобождаем память
    del client_activity
    gc.collect()

# 4. СОЗДАНИЕ КЛЮЧЕВЫХ ПРИЗНАКОВ
print("\n🔧 Шаг 4: Создание ключевых признаков клиентов...")

# Только самые важные признаки для экономии памяти и скорости
print("  📊 Базовые метрики...")
basic_features = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'std', 'count', 'median'],
    'transaction_timestamp': ['min', 'max'],
    'merchant_id': 'nunique',
    'merchant_city': 'nunique'
}).reset_index()

basic_features.columns = ['card_id', 'total_amount', 'avg_amount', 'std_amount', 
                         'transaction_count', 'median_amount', 'first_transaction', 
                         'last_transaction', 'unique_merchants', 'unique_cities']

print("  ⏰ Временные паттерны...")
time_features = df.groupby('card_id').agg({
    'hour': 'mean',
    'is_weekend': 'mean',
    'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
}).reset_index()

time_features.columns = ['card_id', 'avg_hour', 'weekend_ratio', 'preferred_day']

print("  🏪 Категории и типы...")
category_features = df.groupby('card_id').agg({
    'mcc_category': 'nunique',
    'transaction_type': 'nunique'
}).reset_index()

category_features.columns = ['card_id', 'unique_categories', 'unique_txn_types']

# Объединяем признаки
client_features = basic_features.merge(time_features, on='card_id', how='left')
client_features = client_features.merge(category_features, on='card_id', how='left')

# Вычисляемые признаки
client_features['activity_days'] = (client_features['last_transaction'] - 
                                   client_features['first_transaction']).dt.days + 1
client_features['avg_daily_transactions'] = client_features['transaction_count'] / client_features['activity_days']
client_features['coefficient_variation'] = client_features['std_amount'] / client_features['avg_amount']
client_features['avg_monthly_amount'] = client_features['total_amount'] / (client_features['activity_days'] / 30)

# Заполнение пропусков и приведение к float
print("🔧 Финальная обработка признаков...")
client_features = client_features.fillna(0)
client_features['coefficient_variation'] = client_features['coefficient_variation'].replace([np.inf, -np.inf], 0)

# Принудительное приведение всех числовых столбцов к float
numeric_cols = client_features.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'card_id':
        client_features[col] = pd.to_numeric(client_features[col], errors='coerce').fillna(0).astype(float)

print("✅ Все числовые столбцы приведены к float64")

# Освобождение памяти
del df
gc.collect()

print(f"✅ Создано {len(client_features.columns)-1} признаков для {len(client_features):,} клиентов")

# 5. ПОДГОТОВКА К КЛАСТЕРИЗАЦИИ
print("\n⚙️ Шаг 5: Подготовка к кластеризации...")

# Выбираем только самые важные признаки
key_features = [
    'total_amount', 'avg_amount', 'transaction_count', 'activity_days',
    'unique_merchants', 'unique_cities', 'weekend_ratio', 'avg_hour',
    'unique_categories', 'coefficient_variation', 'avg_daily_transactions'
]

print(f"Используем {len(key_features)} ключевых признаков")

# Подготовка матрицы признаков
X = client_features[key_features].copy()

# Принудительное приведение к float ПЕРЕД любыми операциями
print("🔧 Приведение всех данных к float...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

# Простая обработка выбросов (только экстремальные значения)
print("🔧 Обработка выбросов...")
for col in X.columns:
    Q99 = float(X[col].quantile(0.99))
    Q01 = float(X[col].quantile(0.01))
    
    # Безопасная замена выбросов
    X.loc[X[col] > Q99, col] = Q99
    X.loc[X[col] < Q01, col] = Q01

# Заменяем inf и nan
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Финальная проверка типов
print("✅ Финальная проверка типов...")
for col in X.columns:
    if X[col].dtype != 'float64':
        print(f"  Исправляем {col}: {X[col].dtype} → float64")
        X[col] = X[col].astype(float)

print(f"Матрица признаков: {X.shape}")
print(f"Типы данных: {X.dtypes.unique()}")

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Данные подготовлены к кластеризации")

# 6. ОПТИМИЗИРОВАННАЯ КЛАСТЕРИЗАЦИЯ
print("\n🎯 Шаг 6: Кластеризация (только K-Means для стабильности)...")

n_clients = len(X_scaled)
print(f"Клиентов для кластеризации: {n_clients:,}")

# Поиск оптимального количества кластеров
print("📊 Поиск оптимального количества кластеров...")

K_range = range(3, 9)
inertias = []
silhouette_scores = []

for k in K_range:
    print(f"  Тестируем K={k}...", end="")
    
    # Быстрая кластеризация
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    labels = kmeans.fit_predict(X_scaled)
    
    # Метрики
    inertias.append(kmeans.inertia_)
    
    # Силуэт на выборке для скорости
    if n_clients > 5000:
        sample_size = 3000
        sample_indices = np.random.choice(n_clients, sample_size, replace=False)
        sil_score = silhouette_score(X_scaled[sample_indices], labels[sample_indices])
    else:
        sil_score = silhouette_score(X_scaled, labels)
    
    silhouette_scores.append(sil_score)
    print(f" силуэт: {sil_score:.3f}")

# Выбор оптимального K
optimal_k = K_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)

print(f"\n✅ Оптимальное K: {optimal_k} (силуэт: {best_silhouette:.3f})")

# Если получилось мало кластеров, принудительно увеличиваем
if optimal_k < 5:
    optimal_k = 6
    print(f"🔧 Принудительно увеличиваем до {optimal_k} кластеров")

# Финальная кластеризация
print(f"🎯 Финальная кластеризация с K={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# Добавляем результаты
client_features['cluster'] = final_labels

print(f"✅ Кластеризация завершена: {optimal_k} кластеров")

# 7. БЫСТРЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ
print("\n📈 Шаг 7: Анализ кластеров...")

print("📊 Распределение клиентов по кластерам:")
cluster_distribution = pd.Series(final_labels).value_counts().sort_index()
for cluster_id, count in cluster_distribution.items():
    percentage = count / len(client_features) * 100
    print(f"  Кластер {cluster_id}: {count:,} клиентов ({percentage:.1f}%)")

# Профили кластеров (только ключевые метрики)
print(f"\n💡 Профили кластеров:")

key_profile_features = ['total_amount', 'avg_amount', 'transaction_count', 
                       'unique_merchants', 'weekend_ratio', 'avg_hour']

for cluster_id in sorted(client_features['cluster'].unique()):
    cluster_data = client_features[client_features['cluster'] == cluster_id]
    size = len(cluster_data)
    
    print(f"\n🔹 КЛАСТЕР {cluster_id} ({size:,} клиентов):")
    
    # Топ-3 характеристики
    total_avg = cluster_data['total_amount'].mean()
    txn_avg = cluster_data['transaction_count'].mean()
    merchants_avg = cluster_data['unique_merchants'].mean()
    weekend_ratio = cluster_data['weekend_ratio'].mean()
    
    print(f"  • Общая сумма: {total_avg:,.0f} тенге")
    print(f"  • Транзакций: {txn_avg:.0f}")
    print(f"  • Продавцов: {merchants_avg:.1f}")
    print(f"  • Weekend активность: {weekend_ratio:.1%}")

# 8. ПРОСТАЯ ВИЗУАЛИЗАЦИЯ
print("\n🎨 Шаг 8: Визуализация...")

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Простые графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Анализ {optimal_k} кластеров клиентов', fontsize=14)

# 1. PCA визуализация
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    mask = final_labels == i
    axes[0,0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                     c=[colors[i]], label=f'Кластер {i}', alpha=0.6, s=20)

axes[0,0].set_title('Кластеры в пространстве признаков')
axes[0,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
axes[0,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
axes[0,0].legend()

# 2. Размеры кластеров
cluster_sizes = [sum(final_labels == i) for i in range(optimal_k)]
axes[0,1].bar(range(optimal_k), cluster_sizes, color=colors)
axes[0,1].set_title('Размеры кластеров')
axes[0,1].set_xlabel('Номер кластера')
axes[0,1].set_ylabel('Количество клиентов')

# 3. Общие суммы по кластерам
cluster_means = client_features.groupby('cluster')['total_amount'].mean()
axes[1,0].bar(cluster_means.index, cluster_means.values, color=colors)
axes[1,0].set_title('Средние общие суммы')
axes[1,0].set_xlabel('Номер кластера')
axes[1,0].set_ylabel('Средняя сумма (тенге)')
axes[1,0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 4. Активность по кластерам
cluster_activity = client_features.groupby('cluster')['transaction_count'].mean()
axes[1,1].bar(cluster_activity.index, cluster_activity.values, color=colors)
axes[1,1].set_title('Средняя активность')
axes[1,1].set_xlabel('Номер кластера')
axes[1,1].set_ylabel('Среднее кол-во транзакций')

plt.tight_layout()
plt.show()

# График оптимального K
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_title('Метод локтя')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('Инерция')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_title('Силуэт анализ')
ax2.set_xlabel('Количество кластеров')  
ax2.set_ylabel('Силуэт коэффициент')
ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
ax2.grid(True)

plt.tight_layout()
plt.show()

# 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\n💾 Шаг 9: Сохранение результатов...")

# Основной файл результатов
output_columns = ['card_id', 'total_amount', 'avg_amount', 'transaction_count', 
                 'activity_days', 'unique_merchants', 'unique_cities', 
                 'weekend_ratio', 'avg_hour', 'unique_categories', 'cluster']

final_results = client_features[output_columns].copy()
final_results.to_csv('lightweight_client_segments.csv', index=False)
print("✅ Результаты сохранены в 'lightweight_client_segments.csv'")

# Профили кластеров
cluster_profiles = client_features.groupby('cluster')[key_features].agg(['mean', 'median']).round(2)
cluster_profiles.to_csv('cluster_profiles_summary.csv')
print("✅ Профили кластеров сохранены в 'cluster_profiles_summary.csv'")

# Краткая сводка
summary = {
    'total_clients_analyzed': len(client_features),
    'clusters_found': optimal_k,
    'silhouette_score': best_silhouette,
    'features_used': len(key_features),
    'largest_cluster_size': max(cluster_sizes),
    'smallest_cluster_size': min(cluster_sizes)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('analysis_summary.csv', index=False)
print("✅ Сводка анализа сохранена в 'analysis_summary.csv'")

print(f"\n🎉 ОБЛЕГЧЕННЫЙ АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 55)
print(f"📊 ИТОГОВАЯ СТАТИСТИКА:")
print(f"• Проанализировано клиентов: {len(client_features):,}")
print(f"• Использовано признаков: {len(key_features)}")
print(f"• Найдено кластеров: {optimal_k}")
print(f"• Качество кластеризации: {best_silhouette:.3f}")
print(f"• Самый большой кластер: {max(cluster_sizes):,} клиентов")
print(f"• Самый маленький кластер: {min(cluster_sizes):,} клиентов")

balance_ratio = min(cluster_sizes) / max(cluster_sizes)
if balance_ratio > 0.1:
    print("✅ Кластеры относительно сбалансированы")
elif balance_ratio > 0.05:
    print("⚠️ Есть дисбаланс в размерах кластеров")
else:
    print("❌ Сильный дисбаланс - один доминирующий кластер")

print(f"\n📁 Созданные файлы:")
print("• lightweight_client_segments.csv - основные результаты")
print("• cluster_profiles_summary.csv - профили кластеров")
print("• analysis_summary.csv - сводка анализа")
print("\n⚡ Анализ оптимизирован для больших данных и стабильной работы!")
print("🎯 Готово для интерпретации и бизнес-решений!")