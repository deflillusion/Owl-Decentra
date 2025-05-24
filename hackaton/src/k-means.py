from calendar import c
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

data_path = r"C:\Users\ksyus\Desktop\Education\Owl-Decentra\hackaton\data"  # Укажите путь к папке, содержащей parquet-файлы
parquet_files = []

if os.path.exists(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.parquet'):
            parquet_files.append(os.path.join(data_path, filename))
            print(f"  📁 Найден: {filename}")
else:
    print(f"❌ Папка данных не найдена: {data_path}")
    print("🚫 Данные не загружены, завершаем анализ")
    exit()

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
else:
    print("❌ Не найдено ни одного parquet-файла в папке данных.")
    print("🚫 Данные не загружены, завершаем анализ")
    exit()

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

# Временные паттерны (без изменений)
print("  ⏰ Временные паттерны...")
time_features = df.groupby('card_id').agg({
    'hour': 'mean',
    'is_weekend': 'mean',
    'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 0
}).reset_index()

time_features.columns = ['card_id', 'avg_hour', 'weekend_ratio', 'preferred_day']

# Категории и типы (добавляем новые признаки по категориям и типам транзакций)
print("  🏪 Категории и типы...")
# ТЮНИНГ: Определите MCC-коды и типы транзакций
AUTO_MCC = [5541, 5542, 7513, 7531, 5599]  # Удалил дублирующийся '7531'
CASH_WITHDRAWAL_TYPES = ['ATM_WITHDRAWAL']  # Типы транзакций для снятия наличных
INCOMING_TRANSFER_TYPES = ['P2P_IN']  # Типы входящих переводов
OUTGOING_TRANSFER_TYPES = ['P2P_OUT']  # Типы исходящих переводов
SALARY_PAYMENT_TYPES = ['SALARY']  # Типы зарплатных платежей
ECOM_PAYMENT_TYPES = ['ECOM']  # Типы онлайн-платежей
POS_PAYMENT_TYPES = ['POS']  # Типы платежей в магазинах
TRANSACTION_CURRENCY = ['KZT', 'TRY', 'CNY', 'AED', 'AMD', 'BYN', 'KGS', 'UZS', 'USD', 'GEL', 'EUR']  # Валюта транзакций
COUNTRY_ISO = ['KAZ', 'TUR', 'CHN', 'ARE', 'ARM', 'BLR', 'KGZ', 'UZB', 'USA', 'GEO', 'ITA']  # ISO-коды стран
TRAVEL_MCC = [7011, 4111, 4789, 3000]  # Путешествия
REUSTORANT_MCC = [5812, 5813, 5814]  # Рестораны и кафе
COSMETIX_MCC = [5999, 7298, 5945, 5977]  # Косметика
FASHION_MCC = [5651, 5699, 5691]  # Мода
PRODUCT_MCC = [5411, 5499, 5462, 5300]  # Продуктовые
BEAUTY_SALONS_MCC = [7230]  # Салоны красоты
CONSTRUCTION_MCC = [5311, 5310, 5122, 5712, 5200, 5211, 5722, 5732, 5734]  # Добавлены стройматериалы, техника, электроника
COMMUNICATION_AND_INTERNET_MCC = [4814, 4900]  # Связь и интернет
TAX_PAYMENT_MCC = [9311]  # Налоги
DRUG_STORE_MCC = [5912]  # Аптеки
LEGAL_SERVICES_MCC = [8111]  # Юристы
BOOK_AND_SPORTS_MCC = [5941, 5942]  # Книги и спорттовары
PROFESSIONAL_SERVICES_MCC = [8999, 8011]  # Прочие проф.услуги и медицина
MISSING_MCC = [None]  # <NA>, отсутствующий MCC

# Подсчет доли транзакций E-commerce
ecom_spending = df[df['transaction_type'].isin(ECOM_PAYMENT_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
ecom_spending.columns = ['card_id', 'ecom_spending_amount', 'ecom_transaction_count']

# Подсчет доли транзакций в POS-терминалах
pos_spending = df[df['transaction_type'].isin(POS_PAYMENT_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
pos_spending.columns = ['card_id', 'pos_spending_amount', 'pos_transaction_count']

# Подсчет доли трат в автосервисах
auto_spending = df[df['merchant_mcc'].isin(AUTO_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
auto_spending.columns = ['card_id', 'auto_spending_amount', 'auto_transaction_count']

salary_spending = df[df['transaction_type'].isin(SALARY_PAYMENT_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
salary_spending.columns = ['card_id', 'salary_spending_amount', 'salary_transaction_count']

# Подсчет доли снятий наличных
cash_withdrawals = df[df['transaction_type'].isin(CASH_WITHDRAWAL_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
cash_withdrawals.columns = ['card_id', 'cash_withdrawal_amount', 'cash_withdrawal_count']

# Подсчет доли входящих переводов
incoming_transfers = df[df['transaction_type'].isin(INCOMING_TRANSFER_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
incoming_transfers.columns = ['card_id', 'incoming_transfer_amount', 'incoming_transfer_count']

# Подсчет доли исходящих переводов
outcoming_transfers = df[df['transaction_type'].isin(OUTGOING_TRANSFER_TYPES)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
outcoming_transfers.columns = ['card_id', 'outcoming_transfer_amount', 'outcoming_transfer_count']

# Подсчет доли трат в ресторанах
restaurant_spending = df[df['merchant_mcc'].isin(REUSTORANT_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
restaurant_spending.columns = ['card_id', 'restaurant_spending_amount', 'restaurant_transaction_count']

# Подсчет доли трат на косметику
cosmetic_spending = df[df['merchant_mcc'].isin(COSMETIX_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
cosmetic_spending.columns = ['card_id', 'cosmetic_spending_amount', 'cosmetic_transaction_count']

# Подсчет доли трат на моду
fashion_spending = df[df['merchant_mcc'].isin(FASHION_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
fashion_spending.columns = ['card_id', 'fashion_spending_amount', 'fashion_transaction_count']

# Подсчет доли трат на продукты
product_spending = df[df['merchant_mcc'].isin(PRODUCT_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
product_spending.columns = ['card_id', 'product_spending_amount', 'product_transaction_count']

# Подсчет доли трат в салонах красоты
beauty_salons_spending = df[df['merchant_mcc'].isin(BEAUTY_SALONS_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
beauty_salons_spending.columns = ['card_id', 'beauty_salons_spending_amount', 'beauty_salons_transaction_count']

# Подсчет доли трат на строительство
construction_spending = df[df['merchant_mcc'].isin(CONSTRUCTION_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
construction_spending.columns = ['card_id', 'construction_spending_amount', 'construction_transaction_count']

# Подсчет доли трат на связь и интернет
communication_and_internet_spending = df[df['merchant_mcc'].isin(COMMUNICATION_AND_INTERNET_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
communication_and_internet_spending.columns = ['card_id', 'communication_and_internet_spending_amount', 'communication_and_internet_transaction_count']

# Подсчет доли налоговых платежей
tax_payment_spending = df[df['merchant_mcc'].isin(TAX_PAYMENT_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
tax_payment_spending.columns = ['card_id', 'tax_payment_spending_amount', 'tax_payment_transaction_count']

# Подсчет доли трат в аптеках
drug_store_spending = df[df['merchant_mcc'].isin(DRUG_STORE_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
drug_store_spending.columns = ['card_id', 'drug_store_spending_amount', 'drug_store_transaction_count']

# Подсчет доли трат на путешествия
travel_spending = df[df['merchant_mcc'].isin(TRAVEL_MCC)].groupby('card_id').agg({
    'transaction_amount_kzt': 'sum',
    'transaction_id': 'count'
}).reset_index()
travel_spending.columns = ['card_id', 'travel_spending_amount', 'travel_transaction_count']


# Объединяем признаки
client_features = basic_features.merge(time_features, on='card_id', how='left')
client_features = client_features.merge(auto_spending, on='card_id', how='left')
client_features = client_features.merge(cash_withdrawals, on='card_id', how='left')
client_features = client_features.merge(incoming_transfers, on='card_id', how='left')
client_features = client_features.merge(outcoming_transfers, on='card_id', how='left')
client_features = client_features.merge(restaurant_spending, on='card_id', how='left')
client_features = client_features.merge(cosmetic_spending, on='card_id', how='left')
client_features = client_features.merge(fashion_spending, on='card_id', how='left')
client_features = client_features.merge(product_spending, on='card_id', how='left')
client_features = client_features.merge(beauty_salons_spending, on='card_id', how='left')
client_features = client_features.merge(construction_spending, on='card_id', how='left')
client_features = client_features.merge(communication_and_internet_spending, on='card_id', how='left')
client_features = client_features.merge(tax_payment_spending, on='card_id', how='left')
client_features = client_features.merge(drug_store_spending, on='card_id', how='left')
client_features = client_features.merge(travel_spending, on='card_id', how='left')
client_features = client_features.merge(salary_spending, on='card_id', how='left')
client_features = client_features.merge(ecom_spending, on='card_id', how='left')
client_features = client_features.merge(pos_spending, on='card_id', how='left')
foreign_txn = df.groupby('card_id')['acquirer_country_iso'].apply(lambda x: any(x != 'KAZ')).reset_index()
foreign_txn.columns = ['card_id', 'has_foreign_txn']
client_features = client_features.merge(foreign_txn, on='card_id', how='left')
client_features['has_foreign_txn'] = client_features['has_foreign_txn'].fillna(False)



# Заполняем пропуски для новых признаков
client_features[['auto_spending_amount', 'auto_transaction_count']] = client_features[['auto_spending_amount', 'auto_transaction_count']].fillna(0)
client_features[['cash_withdrawal_amount', 'cash_withdrawal_count']] = client_features[['cash_withdrawal_amount', 'cash_withdrawal_count']].fillna(0)
client_features[['incoming_transfer_amount', 'incoming_transfer_count']] = client_features[['incoming_transfer_amount', 'incoming_transfer_count']].fillna(0)
client_features[['outcoming_transfer_amount', 'outcoming_transfer_count']] = client_features[['outcoming_transfer_amount', 'outcoming_transfer_count']].fillna(0)
client_features[['restaurant_spending_amount', 'restaurant_transaction_count']] = client_features[['restaurant_spending_amount', 'restaurant_transaction_count']].fillna(0)
client_features[['cosmetic_spending_amount', 'cosmetic_transaction_count']] = client_features[['cosmetic_spending_amount', 'cosmetic_transaction_count']].fillna(0)
client_features[['fashion_spending_amount', 'fashion_transaction_count']] = client_features[['fashion_spending_amount', 'fashion_transaction_count']].fillna(0)
client_features[['product_spending_amount', 'product_transaction_count']] = client_features[['product_spending_amount', 'product_transaction_count']].fillna(0)
client_features[['beauty_salons_spending_amount', 'beauty_salons_transaction_count']] = client_features[['beauty_salons_spending_amount', 'beauty_salons_transaction_count']].fillna(0)
client_features[['construction_spending_amount', 'construction_transaction_count']] = client_features[['construction_spending_amount', 'construction_transaction_count']].fillna(0)
client_features[['communication_and_internet_spending_amount', 'communication_and_internet_transaction_count']] = client_features[['communication_and_internet_spending_amount', 'communication_and_internet_transaction_count']].fillna(0)
client_features[['tax_payment_spending_amount', 'tax_payment_transaction_count']] = client_features[['tax_payment_spending_amount', 'tax_payment_transaction_count']].fillna(0)
client_features[['drug_store_spending_amount', 'drug_store_transaction_count']] = client_features[['drug_store_spending_amount', 'drug_store_transaction_count']].fillna(0)
client_features[['travel_spending_amount', 'travel_transaction_count']] = client_features[['travel_spending_amount', 'travel_transaction_count']].fillna(0)
client_features[['salary_spending_amount', 'salary_transaction_count']] = client_features[['salary_spending_amount', 'salary_transaction_count']].fillna(0)
client_features[['ecom_spending_amount', 'ecom_transaction_count']] = client_features[['ecom_spending_amount', 'ecom_transaction_count']].fillna(0)
client_features[['pos_spending_amount', 'pos_transaction_count']] = client_features[['pos_spending_amount', 'pos_transaction_count']].fillna(0)

# Вычисляем доли от общей суммы и количества транзакций
client_features['auto_spending_ratio'] = client_features['auto_spending_amount'] / client_features['total_amount']
client_features['cash_withdrawal_ratio'] = client_features['cash_withdrawal_amount'] / client_features['total_amount']
client_features['incoming_transfer_ratio'] = client_features['incoming_transfer_amount'] / client_features['total_amount']
client_features['auto_transaction_ratio'] = client_features['auto_transaction_count'] / client_features['transaction_count']
client_features['cash_withdrawal_transaction_ratio'] = client_features['cash_withdrawal_count'] / client_features['transaction_count']
client_features['incoming_transfer_transaction_ratio'] = client_features['incoming_transfer_count'] / client_features['transaction_count']
client_features['outcoming_transfer_ratio'] = client_features['outcoming_transfer_amount'] / client_features['total_amount']
client_features['restaurant_spending_ratio'] = client_features['restaurant_spending_amount'] / client_features['total_amount']
client_features['cosmetic_spending_ratio'] = client_features['cosmetic_spending_amount'] / client_features['total_amount']
client_features['fashion_spending_ratio'] = client_features['fashion_spending_amount'] / client_features['total_amount']
client_features['product_spending_ratio'] = client_features['product_spending_amount'] / client_features['total_amount']
client_features['beauty_salons_spending_ratio'] = client_features['beauty_salons_spending_amount'] / client_features['total_amount']
client_features['construction_spending_ratio'] = client_features['construction_spending_amount'] / client_features['total_amount']
client_features['communication_and_internet_spending_ratio'] = client_features['communication_and_internet_spending_amount'] / client_features['total_amount']
client_features['tax_payment_spending_ratio'] = client_features['tax_payment_spending_amount'] / client_features['total_amount']
client_features['drug_store_spending_ratio'] = client_features['drug_store_spending_amount'] / client_features['total_amount']
client_features['travel_spending_ratio'] = client_features['travel_spending_amount'] / client_features['total_amount']
client_features['salary_spending_ratio'] = client_features['salary_spending_amount'] / client_features['total_amount']
client_features['ecom_spending_ratio'] = client_features['ecom_spending_amount'] / client_features['total_amount']
client_features['pos_spending_ratio'] = client_features['pos_spending_amount'] / client_features['total_amount']

# ТЮНИНГ: Можете добавить другие категории (например, супермаркеты, путешествия) аналогичным образом
# Например:
# SUPERMARKET_MCC = ['5411']  # MCC для супермаркетов
# supermarket_spending = df[df['mcc_category'].isin(SUPERMARKET_MCC)].groupby('card_id').agg(...)

# Заполняем пропуски и обрабатываем бесконечности
client_features = client_features.fillna(0)
client_features[['auto_spending_ratio', 'cash_withdrawal_ratio', 'incoming_transfer_ratio', 
                 'auto_transaction_ratio', 'cash_withdrawal_transaction_ratio', 
                 'incoming_transfer_transaction_ratio', 'outcoming_transfer_ratio',
                 'restaurant_spending_ratio', 'cosmetic_spending_ratio',
                 'fashion_spending_ratio', 'product_spending_ratio', 'beauty_salons_spending_ratio',
                 'construction_spending_ratio', 'communication_and_internet_spending_ratio',
                 'tax_payment_spending_ratio', 'drug_store_spending_ratio', 'travel_spending_ratio', 'pos_spending_ratio']] = client_features[['auto_spending_ratio', 'cash_withdrawal_ratio', 'incoming_transfer_ratio', 
                 'auto_transaction_ratio', 'cash_withdrawal_transaction_ratio', 
                 'incoming_transfer_transaction_ratio', 'outcoming_transfer_ratio',
                 'restaurant_spending_ratio', 'cosmetic_spending_ratio',
                 'fashion_spending_ratio', 'product_spending_ratio', 'beauty_salons_spending_ratio',
                 'construction_spending_ratio', 'communication_and_internet_spending_ratio',
                 'tax_payment_spending_ratio', 'drug_store_spending_ratio', 'travel_spending_ratio', 'pos_spending_ratio']].replace([np.inf, -np.inf], 0)

# Вычисляемые признаки (без изменений)
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

# ТЮНИНГ: Выберите, какие признаки включить в кластеризацию
key_features = [
    'total_amount', 'avg_amount', 'transaction_count', 'activity_days',
    'unique_merchants', 'unique_cities', 'weekend_ratio', 'avg_hour',
    'auto_spending_ratio', 'cash_withdrawal_ratio', 'incoming_transfer_ratio', 
    'auto_transaction_ratio', 'cash_withdrawal_transaction_ratio', 
    'incoming_transfer_transaction_ratio', 'outcoming_transfer_ratio',
    'restaurant_spending_ratio', 'cosmetic_spending_ratio',
    'fashion_spending_ratio', 'product_spending_ratio', 'beauty_salons_spending_ratio',
    'construction_spending_ratio', 'communication_and_internet_spending_ratio',
    'tax_payment_spending_ratio', 'drug_store_spending_ratio', 'travel_spending_ratio',
    'salary_spending_ratio', 'ecom_spending_ratio', 'pos_spending_ratio'
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
    
    # ТЮНИНГ: Настройте пороги выбросов, если нужно
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

# Профили кластеров
print(f"\n💡 Профили кластеров:")

key_profile_features = ['total_amount', 'avg_amount', 'transaction_count', 
                       'unique_merchants', 'weekend_ratio', 'avg_hour',
                       'auto_spending_ratio', 'cash_withdrawal_ratio', 'incoming_transfer_ratio', 
                       'auto_transaction_ratio', 'cash_withdrawal_transaction_ratio', 
                       'incoming_transfer_transaction_ratio', 'outcoming_transfer_ratio',
                       'restaurant_spending_ratio', 'cosmetic_spending_ratio',
                       'fashion_spending_ratio', 'product_spending_ratio', 'beauty_salons_spending_ratio',
                       'construction_spending_ratio', 'communication_and_internet_spending_ratio',
                       'tax_payment_spending_ratio', 'drug_store_spending_ratio', 'travel_spending_ratio',
                       'salary_spending_ratio', 'ecom_spending_ratio', 'pos_spending_ratio']

# Функция для интерпретации кластера
def interpret_cluster(cluster_data):
    # Извлекаем средние значения признаков
    auto_ratio = cluster_data['auto_spending_ratio'].mean()
    cash_ratio = cluster_data['cash_withdrawal_ratio'].mean()
    transfer_ratio = cluster_data['incoming_transfer_ratio'].mean()
    transaction_count = cluster_data['transaction_count'].mean()
    unique_merchants = cluster_data['unique_merchants'].mean()
    tax_payment_ratio = cluster_data['tax_payment_spending_ratio'].mean()
    weekend_ratio = cluster_data['weekend_ratio'].mean()
    avg_hour = cluster_data['avg_hour'].mean()
    restaurant_ratio = cluster_data['restaurant_spending_ratio'].mean()
    ecom_ratio = cluster_data['ecom_spending_ratio'].mean()
    
    
    # ТЮНИНГ: Настройте пороговые значения для каждого типа клиента

    # Пороги для "Люди использующие наличные"
    cash_threshold = 0.8  # Доля снятий наличных
    tax_threshold = 0.5  # Доля налоговых платежей
    transfer_threshold = 0.5  # Доля входящих переводов
    max_merchants_mo = 200  # Максимальное количество продавцов

    
    # Пороги для "Обычного клиента"
    max_transaction_count = 5000  # Максимальное количество транзакций
    min_merchants = 100  # Минимальное количество продавцов
    
    # Логика классификации
    # Пороги для "Автовладелец"
    auto_threshold = 0.1  # Доля трат на автосервисы
    auto_min_transactions = 10  # Минимальное количество транзакций для "Автолюбителя"
    # Проверка на "Автолюбителя"
    if auto_ratio > auto_threshold and transaction_count >= auto_min_transactions:
        return "Автолюбитель"
    
    # Проверка на "Мошенника"
    elif (cash_ratio > cash_threshold or transfer_ratio > transfer_threshold) and unique_merchants < max_merchants_mo and tax_payment_ratio < tax_threshold:
        return "Потенциальный мошенник"
    
    # Проверка на "Обычного клиента"
    elif transaction_count <= max_transaction_count and unique_merchants >= min_merchants:
        return "Обычный клиент"
    
    elif (weekend_ratio > 0.5 and
        (7 <= avg_hour <= 10 or 12 <= avg_hour <= 14) and
        restaurant_ratio > 0.05):
        return "Офисный работник"
    
    elif ecom_ratio > 0.5:
        return "Удалёнщик"
    
    # ТЮНИНГ: Добавьте новые категории и их критерии
    # Например, для "Путешественника":
    elif cluster_data['has_foreign_txn'].all():
        return "Путешественник (Заграница)"
    
    # Если ни одно условие не выполнено
    return "Неопределенный клиент"

# Вывод профилей кластеров
for cluster_id in sorted(client_features['cluster'].unique()):
    cluster_data = client_features[client_features['cluster'] == cluster_id]
    size = len(cluster_data)
    
    print(f"\n🔹 КЛАСТЕР {cluster_id} ({size:,} клиентов):")
    
    # Интерпретация
    cluster_label = interpret_cluster(cluster_data)
    print(f"  • Тип клиента: {cluster_label}")
    
    # Основные характеристики
    total_avg = cluster_data['total_amount'].mean()
    txn_avg = cluster_data['transaction_count'].mean()
    merchants_avg = cluster_data['unique_merchants'].mean()
    weekend_ratio = cluster_data['weekend_ratio'].mean()
    auto_ratio = cluster_data['auto_spending_ratio'].mean()
    cash_ratio = cluster_data['cash_withdrawal_ratio'].mean()
    transfer_ratio = cluster_data['incoming_transfer_ratio'].mean()
    
    print(f"  • Общая сумма: {total_avg:,.0f} тенге")
    print(f"  • Транзакций: {txn_avg:.0f}")
    print(f"  • Продавцов: {merchants_avg:.1f}")
    print(f"  • Weekend активность: {weekend_ratio:.1%}")
    print(f"  • Доля трат на автосервисы: {auto_ratio:.1%}")
    print(f"  • Доля снятий наличных: {cash_ratio:.1%}")
    print(f"  • Доля входящих переводов: {transfer_ratio:.1%}")
    print(f"  • Доля трат в ресторанах: {cluster_data['restaurant_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на косметику: {cluster_data['cosmetic_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на моду: {cluster_data['fashion_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на продукты: {cluster_data['product_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат в салонах красоты: {cluster_data['beauty_salons_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на стройматериалы, технику, электронику: {cluster_data['construction_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на связь и интернет: {cluster_data['communication_and_internet_spending_ratio'].mean():.1%}")
    print(f"  • Доля налоговых платежей: {cluster_data['tax_payment_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат в аптеках: {cluster_data['drug_store_spending_ratio'].mean():.1%}")
    print(f"  • Доля трат на путешествия: {cluster_data['travel_spending_ratio'].mean():.1%}")
    print(f"  • Зарплата: {cluster_data['salary_spending_ratio'].mean():.1%}")

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
                 'weekend_ratio', 'avg_hour', 'auto_spending_ratio', 
                 'cash_withdrawal_ratio', 'incoming_transfer_ratio', 'cluster']

final_results = client_features[output_columns].copy()
# Добавляем читаемые метки кластеров
final_results['cluster_label'] = final_results['cluster'].apply(
    lambda x: interpret_cluster(client_features[client_features['cluster'] == x])
)

final_results.to_csv('lightweight_client_segments.csv', index=False)
print("✅ Результаты сохранены в 'lightweight_client_segments.csv'")

# Профили кластеров
cluster_profiles = client_features.groupby('cluster')[key_features].agg(['mean', 'median']).round(2)
# Добавляем метку кластера в профили
cluster_profiles['cluster_label'] = [interpret_cluster(client_features[client_features['cluster'] == i]) for i in cluster_profiles.index]
cluster_profiles.to_csv('cluster_profiles_summary.csv')
print("✅ Профили кластеров сохранены в 'cluster_profiles_summary.csv'")