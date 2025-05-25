import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
import gc
warnings.filterwarnings('ignore')

# Определение категорий и типов транзакций
categories = {
    'auto': [5541, 5542, 7513, 7531, 5599],
    'cosmetic': [5999, 7298, 5945, 5977],
    'fashion': [5651, 5699, 5691],
    'beauty_salons': [7230],
    'construction': [5311, 5310, 5122, 5712, 5200, 5211, 5722, 5732, 5734],
    'book_and_sports': [5941, 5942],
    'tax_payment': [9311],
    'travel': [3000, 3010, 3050, 3500, 7011]
}

transaction_types = {
    'incoming_transfer': ['P2P_IN'],
    'outgoing_transfer': ['P2P_OUT'],
    'ecom': ['ECOM'],
    'pos': ['POS'],
    'cash_withdrawal': ['ATM_WITHDRAWAL'],
    'salary': ['SALARY']
}

# Порог для "Путешественника"
min_currencies_travel = 3

# Список для отслеживания использованных меток
used_labels = []

# Настройка для экономии памяти
pd.set_option('display.precision', 2)
plt.style.use('default')

print("🚀 ОБЛЕГЧЕННЫЙ АНАЛИЗ БАНКОВСКИХ ТРАНЗАКЦИЙ")
print("=" * 55)
print("🎯 Цель: Кластеризация клиентов по категориям")
print("⚡ Подход: Умная выборка + оптимизированные признаки")

# 1. ЗАГРУЗКА ДАННЫХ
print("\n📊 Шаг 1: Загрузка данных...")


data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
data_path = os.path.abspath(data_path)
parquet_files = []

if os.path.exists(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.parquet'):
            parquet_files.append(os.path.join(data_path, filename))
            print(f"  📁 Найден: {filename}")
else:
    print(f"❌ Папка данных не найдена: {data_path}")
    exit()

df = None
if parquet_files:
    file_path = parquet_files[0]
    print(f"Загружаем: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(
            f"✅ Данные загружены: {df.shape[0]:,} записей, {df.shape[1]} столбцов")
        print(f"📊 Уникальных клиентов: {df['card_id'].nunique():,}")
        memory_usage = df.memory_usage(deep=True).sum() / 1024**3
        print(f"💾 Размер в памяти: {memory_usage:.1f} GB")
        if memory_usage > 8:
            print("⚠️ Данные очень большие - будем использовать выборку")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        df = None
else:
    print("❌ Не найдено ни одного parquet-файла")
    exit()

if df is None:
    print("🚫 Данные не загружены, завершаем анализ")
    exit()

# 2. БЫСТРАЯ ОЧИСТКА ДАННЫХ
print("\n🧹 Шаг 2: Быстрая очистка данных...")

original_size = len(df)
df = df[df['transaction_amount_kzt'] > 0]
df['transaction_timestamp'] = pd.to_datetime(
    df['transaction_timestamp'], errors='coerce')
df = df[df['transaction_timestamp'].notna()]
print(f"Очистка: {original_size:,} → {len(df):,} записей")

df['hour'] = df['transaction_timestamp'].dt.hour
df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 3. УМНАЯ ВЫБОРКА КЛИЕНТОВ
print("\n🎯 Шаг 3: Создание стратифицированной выборки...")

MAX_CLIENTS = 25000
unique_clients_total = df['card_id'].nunique()
print(f"Всего уникальных клиентов: {unique_clients_total:,}")

if unique_clients_total > MAX_CLIENTS:
    print(f"⚠️ Слишком много клиентов, создаем выборку из {MAX_CLIENTS:,}")
    client_activity = df.groupby('card_id').agg({
        'transaction_amount_kzt': ['count', 'sum'],
        'transaction_timestamp': ['min', 'max']
    })
    client_activity.columns = ['txn_count',
                               'total_amount', 'first_txn', 'last_txn']
    client_activity['activity_days'] = (
        client_activity['last_txn'] - client_activity['first_txn']).dt.days + 1
    client_activity['activity_level'] = pd.cut(
        client_activity['txn_count'],
        bins=[0, 10, 50, 200, float('inf')],
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    sample_clients = []
    for level in ['Low', 'Medium', 'High', 'VeryHigh']:
        level_clients = client_activity[client_activity['activity_level'] == level].index
        sample_size = min(MAX_CLIENTS // 4, len(level_clients))
        if sample_size > 0:
            sample = np.random.choice(
                level_clients, sample_size, replace=False)
            sample_clients.extend(sample)
    remaining = MAX_CLIENTS - len(sample_clients)
    if remaining > 0:
        all_other = client_activity.index.difference(sample_clients)
        if len(all_other) > 0:
            additional = np.random.choice(all_other, min(
                remaining, len(all_other)), replace=False)
            sample_clients.extend(additional)
    df = df[df['card_id'].isin(sample_clients)]
    print(
        f"✅ Выборка: {len(sample_clients):,} клиентов, {len(df):,} транзакций")
    del client_activity
    gc.collect()

# 4. СОЗДАНИЕ КЛЮЧЕВЫХ ПРИЗНАКОВ
print("\n🔧 Шаг 4: Создание ключевых признаков клиентов...")

# Отладка: Проверка данных
print("📋 Колонки в данных:", df.columns.tolist())
print("📋 Уникальные MCC-коды:",
      df['merchant_mcc'].value_counts(dropna=False).head(10))
print("📋 Уникальные типы транзакций:",
      df['transaction_type'].value_counts(dropna=False).head(10))
print("📋 Уникальные валюты:",
      df['transaction_currency'].value_counts(dropna=False).head(10))

# Конвертация merchant_mcc в строки
df['merchant_mcc'] = df['merchant_mcc'].astype(str)

# Базовые метрики
print("  📊 Базовые метрики...")
basic_features = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'std', 'count', 'median'],
    'transaction_timestamp': ['min', 'max'],
    'merchant_id': 'nunique',
    'merchant_city': 'nunique',
    'acquirer_country_iso': lambda x: (x != 'KAZ').any()
}).reset_index()
basic_features.columns = ['card_id', 'total_amount', 'avg_amount', 'std_amount',
                          'transaction_count', 'median_amount', 'first_transaction',
                          'last_transaction', 'unique_merchants', 'unique_cities', 'has_foreign_txn']

# Число уникальных валют
valid_currencies = ['TRY', 'CNY', 'AED', 'AMD',
                    'BYN', 'KGS', 'UZS', 'USD', 'GEL', 'EUR']
currency_features = df[df['transaction_currency'].isin(valid_currencies)].groupby('card_id').agg({
    'transaction_currency': 'nunique'
}).reset_index()
currency_features.columns = ['card_id', 'unique_currencies']
basic_features = basic_features.merge(
    currency_features, on='card_id', how='left')
basic_features['unique_currencies'] = basic_features['unique_currencies'].fillna(
    0).astype(float)

# Временные паттерны
print("  ⏰ Временные паттерны...")
time_features = df.groupby('card_id').agg({
    'hour': 'mean',
    'is_weekend': 'mean',
    'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
}).reset_index()
time_features.columns = ['card_id', 'avg_hour',
                         'weekend_ratio', 'preferred_day']

# Расчет трат по категориям
print("  🏪 Категории и типы...")
client_features = basic_features.merge(time_features, on='card_id', how='left')

for category, mcc_list in categories.items():
    filtered_df = df[df['merchant_mcc'].isin([str(m) for m in mcc_list])]
    print(f"📈 Транзакций для {category}: {len(filtered_df)}")
    category_spending = filtered_df.groupby('card_id').agg({
        'transaction_amount_kzt': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    category_spending.columns = [
        'card_id', f'{category}_spending_amount', f'{category}_transaction_count']
    client_features = client_features.merge(
        category_spending, on='card_id', how='left')

for ttype, type_list in transaction_types.items():
    filtered_df = df[df['transaction_type'].isin(type_list)]
    print(f"📈 Транзакций для {ttype}: {len(filtered_df)}")
    type_spending = filtered_df.groupby('card_id').agg({
        'transaction_amount_kzt': 'sum',
        'transaction_id': 'count'
    }).reset_index()
    type_spending.columns = [
        'card_id', f'{ttype}_spending_amount', f'{ttype}_transaction_count']
    client_features = client_features.merge(
        type_spending, on='card_id', how='left')

# Заполнение пропусков
category_columns = []
for category in categories.keys():
    category_columns.extend(
        [f'{category}_spending_amount', f'{category}_transaction_count'])
for ttype in transaction_types.keys():
    category_columns.extend(
        [f'{ttype}_spending_amount', f'{ttype}_transaction_count'])
client_features[category_columns] = client_features[category_columns].fillna(0)

# Расчет долей
ratio_columns = []
for category in categories.keys():
    client_features[f'{category}_spending_ratio'] = client_features[f'{category}_spending_amount'] / \
        client_features['total_amount']
    client_features[f'{category}_transaction_ratio'] = client_features[f'{category}_transaction_count'] / \
        client_features['transaction_count']
    ratio_columns.extend(
        [f'{category}_spending_ratio', f'{category}_transaction_ratio'])
for ttype in transaction_types.keys():
    client_features[f'{ttype}_spending_ratio'] = client_features[f'{ttype}_spending_amount'] / \
        client_features['total_amount']
    client_features[f'{ttype}_transaction_ratio'] = client_features[f'{ttype}_transaction_count'] / \
        client_features['transaction_count']
    ratio_columns.extend(
        [f'{ttype}_spending_ratio', f'{ttype}_transaction_ratio'])

# Обработка бесконечностей и пропусков
client_features[ratio_columns] = client_features[ratio_columns].replace(
    [np.inf, -np.inf], 0).fillna(0)

# Другие вычисляемые признаки
client_features['activity_days'] = (
    client_features['last_transaction'] - client_features['first_transaction']).dt.days + 1
client_features['avg_daily_transactions'] = client_features['transaction_count'] / \
    client_features['activity_days']
client_features['coefficient_variation'] = client_features['std_amount'] / \
    client_features['avg_amount']
client_features['avg_monthly_amount'] = client_features['total_amount'] / \
    (client_features['activity_days'] / 30)

# Финальная обработка
client_features = client_features.fillna(0)
client_features['coefficient_variation'] = client_features['coefficient_variation'].replace([
                                                                                            np.inf, -np.inf], 0)
numeric_cols = client_features.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'card_id':
        client_features[col] = pd.to_numeric(
            client_features[col], errors='coerce').fillna(0).astype(float)

print("✅ Все числовые столбцы приведены к float64")
del df
gc.collect()
print(
    f"✅ Создано {len(client_features.columns)-1} признаков для {len(client_features):,} клиентов")

# 5. ПОДГОТОВКА К КЛАСТЕРИЗАЦИИ
print("\n⚙️ Шаг 5: Подготовка к кластеризации...")

# Добавляем transaction_count и total_amount в признаки
key_features = [f'{cat}_spending_ratio' for cat in categories.keys()] + \
               [f'{ttype}_spending_ratio' for ttype in transaction_types.keys()] + \
               ['has_foreign_txn', 'unique_currencies',
                   'transaction_count', 'total_amount']
print(f"Используем {len(key_features)} ключевых признаков: {key_features}")

# Увеличиваем веса для ключевых метрик
feature_weights = {
    'auto_spending_ratio': 3.0,
    'cosmetic_spending_ratio': 2.0,
    'fashion_spending_ratio': 2.0,
    'beauty_salons_spending_ratio': 2.0,
    'construction_spending_ratio': 2.5,
    'book_and_sports_spending_ratio': 2.0,
    'tax_payment_spending_ratio': 1.5,
    'travel_spending_ratio': 2.0,
    'incoming_transfer_spending_ratio': 2.0,
    'outgoing_transfer_spending_ratio': 2.0,
    'ecom_spending_ratio': 2.5,
    'pos_spending_ratio': 2.0,
    'cash_withdrawal_spending_ratio': 2.5,
    'salary_spending_ratio': 1.5,
    'has_foreign_txn': 1.5,
    'unique_currencies': 2.5,
    'transaction_count': 2.0,  # Новый вес для количества транзакций
    'total_amount': 2.0        # Новый вес для объёма транзакций
}

X = client_features[key_features].copy()
for feature, weight in feature_weights.items():
    if feature in X.columns:
        X[feature] = X[feature] * weight

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce').astype(float)

print("🔧 Обработка выбросов...")
for col in X.columns:
    Q99 = float(X[col].quantile(0.99))
    Q01 = float(X[col].quantile(0.01))
    X.loc[X[col] > Q99, col] = Q99
    X.loc[X[col] < Q01, col] = Q01

X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

print("✅ Финальная проверка типов...")
for col in X.columns:
    if X[col].dtype != 'float64':
        print(f"  Исправляем {col}: {X[col].dtype} → float64")
        X[col] = X[col].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Данные подготовлены к кластеризации")

# 6. ОПТИМИЗИРОВАННАЯ КЛАСТЕРИЗАЦИЯ
print("\n🎯 Шаг 6: Кластеризация...")

n_clients = len(X_scaled)
print(f"Клиентов для кластеризации: {n_clients:,}")

K_range = range(2, 11)  # Фиксируем 19 кластеров, как в ваших профилях
inertias = []
silhouette_scores = []

for k in K_range:
    print(f"  Тестируем K={k}...", end="")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    if k == 1:
        silhouette_scores.append(float('nan'))
        print(" силуэт: N/A (k=1)")
        continue
    if n_clients > 5000:
        sample_size = 3000
        sample_indices = np.random.choice(
            n_clients, sample_size, replace=False)
        sil_score = silhouette_score(
            X_scaled[sample_indices], labels[sample_indices])
    else:
        sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f" силуэт: {sil_score:.3f}")

optimal_k = K_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)
print(f"\n✅ Оптимальное K: {optimal_k} (силуэт: {best_silhouette:.3f})")

print(f"🎯 Финальная кластеризация с K={optimal_k}...")
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)
client_features['cluster'] = final_labels
print(f"✅ Кластеризация завершена: {optimal_k} кластеров")

# 7. АНАЛИЗ РЕЗУЛЬТАТОВ
print("\n📈 Шаг 7: Анализ кластеров...")

print("📊 Распределение клиентов по кластерам:")
cluster_distribution = pd.Series(final_labels).value_counts().sort_index()
for cluster_id, count in cluster_distribution.items():
    percentage = count / len(client_features) * 100
    print(f"  Кластер {cluster_id}: {count:,} клиентов ({percentage:.1f}%)")

print(f"\n💡 Профили кластеров:")
key_profile_features = ['total_amount', 'transaction_count', 'unique_merchants', 'unique_cities',
                        'weekend_ratio', 'avg_hour', 'has_foreign_txn', 'unique_currencies'] + \
    [f'{cat}_spending_ratio' for cat in categories.keys()] + \
    [f'{ttype}_spending_ratio' for ttype in transaction_types.keys()]


def interpret_cluster(cluster_data, used_labels, cluster_id):
    """
    Определяет тип кластера на основе характеристик клиентов
    Основано на анализе реальных данных из client_segments.csv
    """
    # Базовые метрики
    transaction_count = cluster_data['transaction_count'].mean()
    total_amount = cluster_data['total_amount'].mean(
    ) / 1_000_000  # В миллионах тенге
    unique_currencies = cluster_data['unique_currencies'].mean(
    ) if 'unique_currencies' in cluster_data else 0
    foreign_txn_ratio = cluster_data['has_foreign_txn'].mean(
    ) if 'has_foreign_txn' in cluster_data else 0

    # Категориальные траты (соотношения)
    ratios = {cat: cluster_data[f'{cat}_spending_ratio'].mean()
              for cat in categories.keys()}
    ratios.update({ttype: cluster_data[f'{ttype}_spending_ratio'].mean(
    ) for ttype in transaction_types.keys()})

    # Суммарная доля шопинга (косметика + мода + салоны красоты)
    shopping_ratio = ratios.get(
        'cosmetic', 0) + ratios.get('fashion', 0) + ratios.get('beauty_salons', 0)

    # Отладочная информация
    print(f"\n🔍 Кластер {cluster_id} ({len(cluster_data)} клиентов):")
    print(f"  total_amount: {total_amount:.1f}M₸")
    print(f"  transaction_count: {transaction_count:.0f}")
    print(f"  unique_currencies: {unique_currencies:.1f}")
    print(f"  foreign_txn_ratio: {foreign_txn_ratio:.3f}")
    print(f"  cash_withdrawal_ratio: {ratios.get('cash_withdrawal', 0):.3f}")
    print(f"  ecom_spending_ratio: {ratios.get('ecom', 0):.3f}")
    print(f"  pos_spending_ratio: {ratios.get('pos', 0):.3f}")
    print(f"  shopping_ratio: {shopping_ratio:.3f}")

    # Прямое соответствие на основе анализа данных:

    # Кластер 0: Экономные локальные пользователи
    # Характеристики: 98.1M₸, 3323 транзакций, 0 валют, 88.6% снятие наличных
    if (80 <= total_amount <= 120 and
        3000 <= transaction_count <= 4000 and
        unique_currencies < 0.1 and
            ratios.get('cash_withdrawal', 0) > 0.8):
        return "Экономные локальные пользователи"

    # Кластер 1: Активные премиум-пользователи
    # Характеристики: 119.7M₸, 4946 транзакций, 1.8 валют, 100% зарубежные операции
    elif (100 <= total_amount <= 140 and
          4500 <= transaction_count <= 5500 and
          unique_currencies >= 1.5 and
          foreign_txn_ratio >= 0.9):
        return "Активные премиум-пользователи"

    # Кластер 2: Стабильные локальные покупатели (высокая POS активность)
    # Характеристики: 188.7M₸, 13316 транзакций, 0.1 валют, 72.1% POS
    elif (160 <= total_amount <= 220 and
          transaction_count >= 10000 and
          unique_currencies < 0.5 and
          ratios.get('pos', 0) > 0.6):
        return "Стабильные локальные покупатели"

    # Кластер 3: Глобальные путешественники
    # Характеристики: 191.5M₸, 9281 транзакций, 5.3 валют, 100% зарубежные, 35.7% ecom
    elif (170 <= total_amount <= 220 and
          8000 <= transaction_count <= 11000 and
          unique_currencies >= 4 and
          foreign_txn_ratio >= 0.9 and
          ratios.get('ecom', 0) > 0.3):
        return "Глобальные путешественники"

    # Кластер 4: Пассивные локальные пользователи
    # Характеристики: 6.7M₸, 250 транзакций, 0 валют, 0% зарубежные
    elif (total_amount <= 10 and
          transaction_count <= 300 and
          unique_currencies == 0 and
          foreign_txn_ratio == 0):
        return "Пассивные локальные пользователи"

    # Кластер 5: Спящие локальные пользователи (очень низкая активность)
    # Характеристики: 76.6M₸, 2347 транзакций, 0 валют, 1.5% зарубежные, низкие доли во всех категориях
    elif (60 <= total_amount <= 100 and
          2000 <= transaction_count <= 3000 and
          unique_currencies < 0.1 and
          foreign_txn_ratio < 0.05 and
          ratios.get('ecom', 0) < 0.05 and
          ratios.get('pos', 0) < 0.1):
        return "Спящие локальные пользователи"

    # Кластер 6: Активные автолюбители (или специализированные пользователи)
    # Характеристики: 27.0M₸, 1420 транзакций, 0.5 валют, 37.1% зарубежные, 64.6% ecom
    elif (20 <= total_amount <= 35 and
          1200 <= transaction_count <= 1700 and
          0.3 <= unique_currencies <= 0.7 and
          0.3 <= foreign_txn_ratio <= 0.5 and
          ratios.get('ecom', 0) > 0.6):
        return "Активные автолюбители"

    # Кластер 7: Элитные шопоголики
    # Характеристики: 134.6M₸, 7218 транзакций, 1.4 валют, 50% зарубежные
    elif (120 <= total_amount <= 150 and
          6500 <= transaction_count <= 8000 and
          1.0 <= unique_currencies <= 2.0 and
          0.4 <= foreign_txn_ratio <= 0.6):
        return "Элитные шопоголики"

    # Кластер 8: Стабильные локальные шопоголики
    # Характеристики: 77.4M₸, 3745 транзакций, 0 валют, 1% зарубежные, 33.8% POS, 4.7% шопинг
    elif (65 <= total_amount <= 90 and
          3500 <= transaction_count <= 4000 and
          unique_currencies < 0.1 and
          foreign_txn_ratio < 0.05 and
          0.3 <= ratios.get('pos', 0) <= 0.4 and
          shopping_ratio > 0.04):
        return "Стабильные локальные шопоголики"

    # Кластер 9: Элитные глобальные путешественники
    # Характеристики: 215.8M₸, 8314 транзакций, 0.6 валют, 33.6% зарубежные, 57.3% POS
    elif (total_amount >= 200 and
          transaction_count >= 7000 and
          0.4 <= unique_currencies <= 1.0 and
          0.3 <= foreign_txn_ratio <= 0.4 and
          ratios.get('pos', 0) > 0.5):
        return "Элитные глобальные путешественники"

    # Если не подходит ни под один профиль, используем дополнительную логику
    else:
        # Дополнительная логика на основе общих принципов
        if unique_currencies >= 3 and foreign_txn_ratio >= 0.5:
            return "Глобальные путешественники"
        elif total_amount >= 200 and unique_currencies >= 1:
            return "Элитные глобальные путешественники"
        elif ratios.get('cash_withdrawal', 0) > 0.7:
            return "Экономные локальные пользователи"
        elif ratios.get('ecom', 0) > 0.5:
            return "Активные автолюбители"
        elif transaction_count < 500:
            return "Пассивные локальные пользователи"
        elif total_amount < 30:
            return "Спящие локальные пользователи"
        else:
            return f"Кластер {cluster_id}"


score + global_score + activity_score
return total_priority


# Обновленная логика применения в основном коде
print("🎯 Определение типов кластеров...")

# Получаем приоритеты для всех кластеров
cluster_priorities = []
for cluster_id in sorted(client_features['cluster'].unique()):
    cluster_data = client_features[client_features['cluster'] == cluster_id]
    priority = get_cluster_priority(cluster_data)
    cluster_priorities.append((cluster_id, priority))

# Сортируем по приоритету (от высокого к низкому)
cluster_priorities.sort(key=lambda x: x[1], reverse=True)

# Применяем классификацию к каждому кластеру
used_labels = []  # Сбрасываем список использованных меток
cluster_labels = []

for cluster_id, priority in cluster_priorities:
    cluster_data = client_features[client_features['cluster'] == cluster_id]
    cluster_label = interpret_cluster(cluster_data, used_labels, cluster_id)
    cluster_labels.append((cluster_id, cluster_label))

    # Выводим информацию о кластере
    size = len(cluster_data)
    print(f"\n🔹 КЛАСТЕР {cluster_id} ({size:,} клиентов):")
    print(f"  • Тип клиента: {cluster_label}")
    print(f"  • Приоритет: {priority:.1f}")

    # Основные метрики
    total_avg = cluster_data['total_amount'].mean()
    txn_avg = cluster_data['transaction_count'].mean()
    merchants_avg = cluster_data['unique_merchants'].mean()
    cities_avg = cluster_data['unique_cities'].mean()
    weekend_ratio = cluster_data['weekend_ratio'].mean()
    foreign_txn = cluster_data['has_foreign_txn'].mean()
    unique_currencies = cluster_data['unique_currencies'].mean()

    print(f"  • Общая сумма: {total_avg:,.0f} тенге")
    print(f"  • Транзакций: {txn_avg:.0f}")
    print(f"  • Продавцов: {merchants_avg:.1f}")
    print(f"  • Городов: {cities_avg:.1f}")
    print(f"  • Доля заграничных транзакций: {foreign_txn:.1%}")
    print(f"  • Weekend активность: {weekend_ratio:.1%}")
    print(f"  • Уникальных валют: {unique_currencies:.1f}")

    # Категориальные траты
    for cat in categories.keys():
        ratio = cluster_data[f'{cat}_spending_ratio'].mean()
        if ratio > 0.05:  # Показываем только значимые категории
            print(f"  • Доля трат на {cat}: {ratio:.1%}")

    for ttype in transaction_types.keys():
        ratio = cluster_data[f'{ttype}_spending_ratio'].mean()
        if ratio > 0.1:  # Показываем только значимые типы
            print(f"  • Доля трат на {ttype}: {ratio:.1%}")

# 8. ВИЗУАЛИЗАЦИЯ
print("\n🎨 Шаг 8: Визуализация...")

# Применяем PCA с 3 компонентами
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Получаем метки и метрики для легенды
used_labels = []
cluster_labels = []
cluster_metrics = []
for cluster_id in sorted(client_features['cluster'].unique()):
    cluster_data = client_features[client_features['cluster'] == cluster_id]
    label = interpret_cluster(cluster_data, used_labels, cluster_id)
    avg_total_amount = cluster_data['total_amount'].mean() / 1_000_000
    avg_transaction_count = cluster_data['transaction_count'].mean()
    cluster_labels.append((cluster_id, label))
    cluster_metrics.append(
        (cluster_id, avg_total_amount, avg_transaction_count))

# Нормализация метрик для визуализации
total_amounts = client_features['total_amount'].values
transaction_counts = client_features['transaction_count'].values
amount_min, amount_max = total_amounts.min(), total_amounts.max()
if amount_max > amount_min:
    sizes = 10 + 90 * (total_amounts - amount_min) / (amount_max - amount_min)
else:
    sizes = np.full_like(total_amounts, 50.0)
count_min, count_max = transaction_counts.min(), transaction_counts.max()
if count_max > count_min:
    alphas = 0.3 + 0.5 * (transaction_counts - count_min) / \
        (count_max - count_min)
else:
    alphas = np.full_like(transaction_counts, 0.5)

# 3D-график
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.tab20(np.linspace(0, 1, optimal_k))
for i in range(optimal_k):
    mask = final_labels == i
    cluster_label = next(
        (label for cid, label in cluster_labels if cid == i), f"Кластер {i}")
    avg_amount = next((amt for cid, amt, _ in cluster_metrics if cid == i), 0)
    avg_count = next((cnt for cid, _, cnt in cluster_metrics if cid == i), 0)
    legend_label = f"{cluster_label} (Сумма: {avg_amount:.1f}M₸, Транз: {avg_count:.0f})"
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
               c=[colors[i]], label=legend_label,
               s=sizes[mask], alpha=float(np.mean(alphas[mask])))

ax.set_title('Кластеры в 3D-пространстве PCA с метриками транзакций')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} дисперсии)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig('pca_3d_clusters_with_metrics.png', bbox_inches='tight', dpi=300)
print("✅ 3D-график сохранён как 'pca_3d_clusters_with_metrics.png'")

# Остальные графики
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Анализ {optimal_k} кластеров клиентов', fontsize=14)
cluster_sizes = [sum(final_labels == i) for i in range(optimal_k)]
axes[0, 0].bar(range(optimal_k), cluster_sizes, color=colors)
axes[0, 0].set_title('Размеры кластеров')
axes[0, 0].set_xlabel('Номер кластера')
axes[0, 0].set_ylabel('Количество клиентов')
cluster_travel = client_features.groupby(
    'cluster')['travel_spending_ratio'].mean()
axes[0, 1].bar(cluster_travel.index, cluster_travel.values, color=colors)
axes[0, 1].set_title('Доля трат на путешествия')
axes[0, 1].set_xlabel('Номер кластера')
axes[0, 1].set_ylabel('Средняя доля трат')
cluster_auto = client_features.groupby('cluster')['auto_spending_ratio'].mean()
axes[1, 0].bar(cluster_auto.index, cluster_auto.values, color=colors)
axes[1, 0].set_title('Доля трат на автосервисы')
axes[1, 0].set_xlabel('Номер кластера')
axes[1, 0].set_ylabel('Средняя доля трат')
axes[1, 1].axis('off')
plt.tight_layout()
plt.savefig('cluster_analysis.png', bbox_inches='tight', dpi=300)
print("✅ Анализ кластеров сохранён как 'cluster_analysis.png'")

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
plt.savefig('elbow_silhouette.png', bbox_inches='tight', dpi=300)
print("✅ Метод локтя и силуэт-анализ сохранены как 'elbow_silhouette.png'")
plt.show()

# 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\n💾 Шаг 9: Сохранение результатов...")

output_columns = ['card_id', 'total_amount', 'transaction_count', 'unique_merchants',
                  'unique_cities', 'weekend_ratio', 'avg_hour', 'has_foreign_txn',
                  'unique_currencies', 'cluster'] + \
    [f'{cat}_spending_ratio' for cat in categories.keys()] + \
    [f'{ttype}_spending_ratio' for ttype in transaction_types.keys()]
final_results = client_features[output_columns].copy()
final_results['cluster_label'] = final_results['cluster'].apply(
    lambda x: interpret_cluster(
        client_features[client_features['cluster'] == x], [], x)
)
final_results.to_csv('client_segments.csv', index=False)
print("✅ Результаты сохранены в 'client_segments.csv'")

used_labels_profiles = []
cluster_profiles = client_features.groupby(
    'cluster')[key_profile_features].agg(['mean', 'median']).round(2)
clients_count = client_features.groupby('cluster').size()
cluster_profiles['clients_count'] = clients_count

cluster_profiles.to_csv('cluster_profiles_summary.csv')
print("✅ Профили кластеров сохранены в 'cluster_profiles_summary.csv'")
