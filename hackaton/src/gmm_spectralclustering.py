# ГИБРИДНЫЙ ПОДХОД: ПОЛНЫЙ GMM + КОНТРОЛИРУЕМЫЙ SpectralClustering
# Автор: Erik (Decentra) - Лучшее из двух миров
# Стратегия: Полный GMM анализ + SpectralClustering только для неопределенных

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
import os
import gc
warnings.filterwarnings('ignore')

plt.style.use('default')

print("🎯 ГИБРИДНАЯ КЛАСТЕРИЗАЦИЯ: ПОЛНЫЙ GMM + КОНТРОЛИРУЕМЫЙ SPECTRAL")
print("=" * 70)
print("🔍 GMM: Полный анализ с автовыбором (как в оригинале)")
print("🔍 SpectralClustering: Только для неопределенных клиентов")
print("🚀 Цель: Максимальное качество GMM + точечное использование Spectral")

# === ЗАГРУЗКА ДАННЫХ ===
print("\n📊 Шаг 1: Загрузка данных...")

data_path = "/kaggle/input/decentra"
parquet_files = []

if os.path.exists(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.parquet'):
            parquet_files.append(os.path.join(data_path, filename))

df = None
if parquet_files:
    file_path = parquet_files[0]
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Данные загружены: {df.shape[0]:,} записей")

        # Разумная выборка для качественного GMM
        MAX_CLIENTS = 20000  # Больше для качественного GMM
        unique_clients_total = df['card_id'].nunique()

        if unique_clients_total > MAX_CLIENTS:
            print(f"⚠️ Создаем выборку: {MAX_CLIENTS:,} клиентов")

            # Качественная стратифицированная выборка (как в оригинале)
            client_summary = df.groupby('card_id').agg({
                'transaction_amount_kzt': ['count', 'sum', 'mean', 'std'],
                'transaction_timestamp': ['min', 'max'],
                'merchant_id': 'nunique',
                'mcc_category': 'nunique'
            })

            client_summary.columns = ['txn_count', 'total_amount', 'avg_amount', 'std_amount',
                                      'first_txn', 'last_txn', 'unique_merchants', 'unique_categories']

            # Многомерная стратификация (как в оригинале)
            client_summary['activity_level'] = pd.qcut(
                client_summary['txn_count'], q=5, labels=range(5))
            client_summary['amount_level'] = pd.qcut(
                client_summary['total_amount'], q=4, labels=range(4))
            client_summary['diversity_level'] = pd.qcut(
                client_summary['unique_merchants'], q=3, labels=range(3))

            # Равномерная выборка из каждой страты
            sample_clients = []
            clients_per_strata = MAX_CLIENTS // 60  # 5*4*3 = 60 страт

            for act in range(5):
                for amt in range(4):
                    for div in range(3):
                        strata_clients = client_summary[
                            (client_summary['activity_level'] == act) &
                            (client_summary['amount_level'] == amt) &
                            (client_summary['diversity_level'] == div)
                        ].index

                        sample_size = min(clients_per_strata,
                                          len(strata_clients))
                        if sample_size > 0:
                            sample = np.random.choice(
                                strata_clients, sample_size, replace=False)
                            sample_clients.extend(sample)

            df = df[df['card_id'].isin(sample_clients)]
            print(
                f"✅ Многомерная стратифицированная выборка: {len(sample_clients):,} клиентов")

            del client_summary
            gc.collect()

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        df = None

if df is None:
    print("🚫 Завершаем анализ")
    exit()

# === ПОЛНАЯ ОЧИСТКА И ОБОГАЩЕНИЕ (как в оригинале) ===
print("\n🧹 Шаг 2: Полная очистка и обогащение данных...")

original_size = len(df)
df = df[df['transaction_amount_kzt'] > 0]
df['transaction_timestamp'] = pd.to_datetime(
    df['transaction_timestamp'], errors='coerce')
df = df[df['transaction_timestamp'].notna()]

# Добавляем временные и поведенческие признаки (как в оригинале)
df['hour'] = df['transaction_timestamp'].dt.hour
df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
df['month'] = df['transaction_timestamp'].dt.month
df['quarter'] = df['transaction_timestamp'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

print(f"Очистка: {original_size:,} → {len(df):,} записей")

# === ПОЛНОЕ СОЗДАНИЕ ПРИЗНАКОВ (как в оригинале) ===
print("\n🔧 Шаг 3: Полное создание признаков...")

# Базовые финансовые метрики (как в оригинале)
print("  💰 Финансовые метрики...")
financial_features = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'median', 'std', 'count', 'min', 'max'],
    'transaction_timestamp': ['min', 'max']
}).reset_index()

financial_features.columns = ['card_id', 'total_amount', 'avg_amount', 'median_amount',
                              'std_amount', 'transaction_count', 'min_amount', 'max_amount',
                              'first_transaction', 'last_transaction']

# Поведенческие метрики (как в оригинале)
print("  🛍️ Поведенческие метрики...")
behavioral_features = df.groupby('card_id').agg({
    'merchant_id': 'nunique',
    'merchant_city': 'nunique',
    'mcc_category': 'nunique',
    'transaction_type': 'nunique'
}).reset_index()

behavioral_features.columns = ['card_id', 'unique_merchants', 'unique_cities',
                               'unique_categories', 'unique_txn_types']

# Временные паттерны (как в оригинале)
print("  ⏰ Временные паттерны...")
time_features = df.groupby('card_id').agg({
    'hour': ['mean', 'std'],
    'day_of_week': lambda x: x.mode().iloc[0] if len(x) > 0 else 0,
    'is_weekend': 'mean',
    'is_business_hours': 'mean',
    'is_night': 'mean',
    'month': 'nunique',
    'quarter': 'nunique'
}).reset_index()

time_features.columns = ['card_id', 'avg_hour', 'hour_std', 'preferred_day',
                         'weekend_ratio', 'business_hours_ratio', 'night_ratio',
                         'active_months', 'active_quarters']

# Топ категории MCC (как в оригинале)
print("  🏪 Категории покупок...")
top_categories = df['mcc_category'].value_counts().head(10).index.tolist()
mcc_features = df.groupby('card_id')['mcc_category'].apply(
    lambda x: pd.Series({f'mcc_{cat.lower()}_ratio': (
        x == cat).mean() for cat in top_categories})
).reset_index()

# Продвинутые признаки (как в оригинале)
print("  📊 Продвинутые метрики...")


def calculate_advanced_features(group):
    amounts = group['transaction_amount_kzt']
    timestamps = group['transaction_timestamp']

    # Временные интервалы
    time_diffs = timestamps.diff().dt.total_seconds() / 3600
    time_diffs = time_diffs.dropna()

    return pd.Series({
        'amount_skewness': amounts.skew(),
        'amount_kurtosis': amounts.kurtosis(),
        'amount_range_ratio': (amounts.max() - amounts.min()) / amounts.mean() if amounts.mean() > 0 else 0,
        'large_txn_ratio': (amounts > amounts.quantile(0.9)).mean(),
        'small_txn_ratio': (amounts < amounts.quantile(0.1)).mean(),
        'regularity_score': 1 / (1 + amounts.std() / amounts.mean()) if amounts.mean() > 0 else 0,
        'avg_time_between_txns': time_diffs.mean() if len(time_diffs) > 0 else 0,
        'time_consistency': 1 / (1 + time_diffs.std() / time_diffs.mean()) if len(time_diffs) > 0 and time_diffs.mean() > 0 else 0,
        'txn_density': len(amounts) / ((timestamps.max() - timestamps.min()).days + 1) if len(amounts) > 1 else 0,
        'peak_hour_concentration': group['hour'].value_counts().max() / len(group) if len(group) > 0 else 0
    })


advanced_features = df.groupby('card_id').apply(
    calculate_advanced_features).reset_index()

# Объединяем все признаки (как в оригинале)
print("  🔗 Объединение признаков...")
client_features = financial_features.merge(
    behavioral_features, on='card_id', how='left')
client_features = client_features.merge(
    time_features, on='card_id', how='left')
client_features = client_features.merge(mcc_features, on='card_id', how='left')
client_features = client_features.merge(
    advanced_features, on='card_id', how='left')

# Вычисляемые признаки (как в оригинале)
client_features['activity_days'] = (client_features['last_transaction'] -
                                    client_features['first_transaction']).dt.days + 1
client_features['avg_daily_transactions'] = client_features['transaction_count'] / \
    client_features['activity_days']
client_features['avg_monthly_amount'] = client_features['total_amount'] / \
    (client_features['activity_days'] / 30)
client_features['coefficient_variation'] = client_features['std_amount'] / \
    client_features['avg_amount']
client_features['spending_velocity'] = client_features['total_amount'] / \
    client_features['activity_days']

# Финальная обработка (как в оригинале)
client_features = client_features.fillna(0)
for col in client_features.select_dtypes(include=[np.number]).columns:
    if col != 'card_id':
        client_features[col] = pd.to_numeric(
            client_features[col], errors='coerce').fillna(0).astype(float)

client_features = client_features.replace([np.inf, -np.inf], 0)

# Освобождение памяти
del df
gc.collect()

print(
    f"✅ Создано {len(client_features.columns)-1} признаков для {len(client_features):,} клиентов")

# === ПОДГОТОВКА ДАННЫХ ===
print("\n⚙️ Шаг 4: Подготовка данных...")

# Выбираем признаки (как в оригинале)
feature_cols = client_features.select_dtypes(
    include=[np.number]).columns.tolist()
feature_cols.remove('card_id')

X = client_features[feature_cols].copy()

# Мягкая обработка выбросов (как в оригинале)
for col in X.columns:
    Q98 = float(X[col].quantile(0.98))
    Q02 = float(X[col].quantile(0.02))
    X.loc[X[col] > Q98, col] = Q98
    X.loc[X[col] < Q02, col] = Q02

X = X.astype(float)
print(f"Матрица признаков: {X.shape}")

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ЭТАП 1: ПОЛНЫЙ GMM АНАЛИЗ (как в оригинале) ===
print("\n🎯 ЭТАП 1: Полный GMM анализ...")

n_clients = len(X_scaled)
print(f"Клиентов для анализа: {n_clients:,}")

# Полный поиск оптимального количества компонент (как в оригинале)
n_components_range = range(2, 16)
bic_scores = []
aic_scores = []
silhouette_scores = []

print("📊 Поиск оптимального количества компонент GMM...")
for n_comp in n_components_range:
    print(f"  Тестируем {n_comp} компонент...", end="")

    # Тестируем разные типы ковариационных матриц (как в оригинале)
    best_score = -np.inf
    best_model = None

    for covariance_type in ['full', 'tied', 'diag', 'spherical']:
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=covariance_type,
                random_state=42,
                n_init=3,
                max_iter=200
            )

            gmm.fit(X_scaled)
            score = gmm.score(X_scaled)

            if score > best_score:
                best_score = score
                best_model = gmm
        except:
            continue

    if best_model is not None:
        labels = best_model.predict(X_scaled)

        # Метрики качества (как в оригинале)
        bic = best_model.bic(X_scaled)
        aic = best_model.aic(X_scaled)

        # Силуэт (на выборке для скорости)
        if n_clients > 5000:
            sample_indices = np.random.choice(n_clients, 3000, replace=False)
            sil_score = silhouette_score(
                X_scaled[sample_indices], labels[sample_indices])
        else:
            sil_score = silhouette_score(X_scaled, labels)

        bic_scores.append(bic)
        aic_scores.append(aic)
        silhouette_scores.append(sil_score)

        print(f" BIC: {bic:.0f}, AIC: {aic:.0f}, Силуэт: {sil_score:.3f}")
    else:
        print(" Ошибка")
        bic_scores.append(np.inf)
        aic_scores.append(np.inf)
        silhouette_scores.append(-1)

# Выбор оптимального количества компонент (как в оригинале)
optimal_n_bic = n_components_range[np.argmin(bic_scores)]
optimal_n_aic = n_components_range[np.argmin(aic_scores)]
optimal_n_sil = n_components_range[np.argmax(silhouette_scores)]

print(f"\n📊 Рекомендации по количеству кластеров:")
print(f"• По BIC: {optimal_n_bic} кластеров")
print(f"• По AIC: {optimal_n_aic} кластеров")
print(f"• По силуэту: {optimal_n_sil} кластеров")

# Выбираем компромиссное решение (как в оригинале)
optimal_n_gmm = optimal_n_bic  # BIC обычно лучше для выбора модели
print(f"✅ Выбираем: {optimal_n_gmm} кластеров (по BIC)")

# Финальная GMM модель (как в оригинале)
print("🎯 Обучение финальной GMM модели...")
final_models = {}
for cov_type in ['full', 'tied', 'diag']:
    try:
        gmm = GaussianMixture(
            n_components=optimal_n_gmm,
            covariance_type=cov_type,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        gmm.fit(X_scaled)
        final_models[cov_type] = {
            'model': gmm,
            'bic': gmm.bic(X_scaled),
            'aic': gmm.aic(X_scaled),
            'log_likelihood': gmm.score(X_scaled)
        }

        print(
            f"• {cov_type}: BIC={gmm.bic(X_scaled):.0f}, AIC={gmm.aic(X_scaled):.0f}")
    except Exception as e:
        print(f"• {cov_type}: Ошибка - {e}")

# Выбираем лучшую модель (как в оригинале)
best_cov_type = min(final_models.keys(), key=lambda x: final_models[x]['bic'])
final_gmm = final_models[best_cov_type]['model']

print(f"✅ Лучшая модель: {best_cov_type} covariance")

# Получаем результаты (как в оригинале)
gmm_labels = final_gmm.predict(X_scaled)
gmm_probabilities = final_gmm.predict_proba(X_scaled)
gmm_max_probs = gmm_probabilities.max(axis=1)

# Анализ неопределенности (как в оригинале)
uncertainty_threshold = 0.6
uncertain_clients = (gmm_max_probs < uncertainty_threshold).sum()

print(f"\n📊 Результаты GMM:")
print(f"• Кластеров: {optimal_n_gmm}")
print(f"• Тип ковариации: {best_cov_type}")
print(
    f"• Неопределенных клиентов: {uncertain_clients} ({uncertain_clients/n_clients*100:.1f}%)")

# === ЭТАП 2: КОНТРОЛИРУЕМЫЙ SPECTRAL ДЛЯ НЕОПРЕДЕЛЕННЫХ ===
print("\n🌐 ЭТАП 2: SpectralClustering для неопределенных...")

# Находим неопределенных клиентов
uncertain_mask = gmm_max_probs < uncertainty_threshold
uncertain_count = uncertain_mask.sum()

print(
    f"📊 Неопределенных клиентов GMM: {uncertain_count:,} ({uncertain_count/n_clients*100:.1f}%)")

if uncertain_count > 10:  # Если есть достаточно неопределенных клиентов
    print("🔧 Применяем SpectralClustering к неопределенным клиентам...")

    # Данные только неопределенных клиентов
    X_uncertain = X_scaled[uncertain_mask]

    # Контролируемое количество кластеров для Spectral
    n_spectral_clusters = min(
        5, max(2, uncertain_count // 150))  # Разумное количество

    try:
        spectral = SpectralClustering(
            n_clusters=n_spectral_clusters,
            affinity='rbf',
            random_state=42,
            n_jobs=2  # Ограничиваем параллелизм
        )

        spectral_labels_uncertain = spectral.fit_predict(X_uncertain)
        print(
            f"✅ SpectralClustering: {n_spectral_clusters} микрокластеров для неопределенных")

        # Создаем полный массив меток Spectral
        # -1 для определенных GMM клиентов
        spectral_labels = np.full(n_clients, -1)
        spectral_labels[uncertain_mask] = spectral_labels_uncertain

    except Exception as e:
        print(f"⚠️ SpectralClustering не удался: {e}")
        spectral_labels = np.full(n_clients, -1)
        n_spectral_clusters = 0
else:
    print("⚠️ Слишком мало неопределенных клиентов, пропускаем SpectralClustering")
    spectral_labels = np.full(n_clients, -1)
    n_spectral_clusters = 0

# === СОЗДАНИЕ ГИБРИДНЫХ СЕГМЕНТОВ ===
print("\n🔄 ЭТАП 3: Создание гибридных сегментов...")


def create_hybrid_segments(gmm_label, spectral_label, is_uncertain):
    if is_uncertain and spectral_label != -1:
        return f"REFINED_{spectral_label}"  # Уточненные сегменты
    else:
        return f"MAIN_{gmm_label}"  # Основные GMM сегменты


# Добавляем результаты в датафрейм (как в оригинале)
client_features['gmm_cluster'] = gmm_labels
client_features['gmm_max_prob'] = gmm_max_probs
client_features['gmm_uncertain'] = uncertain_mask
client_features['spectral_cluster'] = spectral_labels

# Добавляем вероятности для каждого кластера (как в оригинале)
for i in range(optimal_n_gmm):
    client_features[f'prob_cluster_{i}'] = gmm_probabilities[:, i]

client_features['hybrid_segment'] = [
    create_hybrid_segments(gmm, spectral, uncertain)
    for gmm, spectral, uncertain in zip(gmm_labels, spectral_labels, uncertain_mask)
]

# Анализ результатов
print("📊 Результаты гибридной сегментации:")
segment_counts = client_features['hybrid_segment'].value_counts()

main_segments = segment_counts[segment_counts.index.str.startswith('MAIN_')]
refined_segments = segment_counts[segment_counts.index.str.startswith(
    'REFINED_')]

print(
    f"  🏛️ Основных GMM сегментов: {len(main_segments)} ({main_segments.sum():,} клиентов)")
if len(refined_segments) > 0:
    print(
        f"  🔬 Уточненных сегментов Spectral: {len(refined_segments)} ({refined_segments.sum():,} клиентов)")

# === ПОЛНЫЙ АНАЛИЗ GMM КЛАСТЕРОВ (как в оригинале) ===
print(f"\n📈 Анализ GMM кластеров:")
print("📊 Распределение клиентов по кластерам:")
cluster_sizes = pd.Series(gmm_labels).value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    percentage = size / n_clients * 100
    avg_prob = gmm_probabilities[gmm_labels == cluster_id, cluster_id].mean()
    print(
        f"  Кластер {cluster_id}: {size:,} клиентов ({percentage:.1f}%), ср.вероятность: {avg_prob:.3f}")

# Профили GMM кластеров (как в оригинале)
print(f"\n💡 Профили GMM кластеров:")
key_metrics = ['total_amount', 'avg_amount', 'transaction_count', 'unique_merchants',
               'weekend_ratio', 'business_hours_ratio', 'regularity_score']

for cluster_id in sorted(pd.Series(gmm_labels).unique()):
    cluster_data = client_features[client_features['gmm_cluster'] == cluster_id]
    size = len(cluster_data)
    avg_certainty = cluster_data['gmm_max_prob'].mean()

    print(
        f"\n🔹 КЛАСТЕР {cluster_id} ({size:,} клиентов, уверенность: {avg_certainty:.3f}):")

    for metric in key_metrics:
        if metric in cluster_data.columns:
            value = cluster_data[metric].mean()
            if 'amount' in metric:
                print(f"  • {metric}: {value:,.0f} тенге")
            elif 'ratio' in metric or 'score' in metric:
                print(f"  • {metric}: {value:.3f}")
            else:
                print(f"  • {metric}: {value:.1f}")

# === ВИЗУАЛИЗАЦИЯ ===
print("\n🎨 Шаг 4: Визуализация...")

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Создание графиков
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    'Гибридная кластеризация: Полный GMM + Контролируемый SpectralClustering', fontsize=16)

# 1. GMM кластеры
colors_gmm = plt.cm.tab10(np.linspace(0, 1, optimal_n_gmm))
for i in range(optimal_n_gmm):
    mask = gmm_labels == i
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors_gmm[i]], label=f'GMM {i}', alpha=0.7, s=15)
axes[0, 0].set_title(f'GMM кластеры ({optimal_n_gmm})')
axes[0, 0].legend()
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

# 2. Карта уверенности GMM
scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_max_probs,
                             cmap='viridis', alpha=0.6, s=15)
axes[0, 1].set_title('GMM: Карта уверенности')
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=axes[0, 1])

# 3. Гибридные сегменты (топ-10)
top_segments = segment_counts.head(10)
colors_hybrid = plt.cm.Set3(np.linspace(0, 1, len(top_segments)))
for i, segment in enumerate(top_segments.index):
    mask = client_features['hybrid_segment'] == segment
    axes[0, 2].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors_hybrid[i]], label=segment, alpha=0.7, s=15)
axes[0, 2].set_title('Гибридные сегменты (топ-10)')
axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[0, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

# 4. BIC кривая GMM (как в оригинале)
axes[1, 0].plot(n_components_range, bic_scores, 'b-o', label='BIC')
axes[1, 0].plot(n_components_range, aic_scores, 'r-s', label='AIC')
axes[1, 0].axvline(x=optimal_n_gmm, color='green',
                   linestyle='--', label=f'Выбрано: {optimal_n_gmm}')
axes[1, 0].set_title('Критерии выбора модели')
axes[1, 0].set_xlabel('Количество кластеров')
axes[1, 0].set_ylabel('Значение критерия')
axes[1, 0].legend()

# 5. Силуэт анализ (как в оригинале)
axes[1, 1].plot(n_components_range, silhouette_scores, 'g-^')
axes[1, 1].axvline(x=optimal_n_gmm, color='green', linestyle='--')
axes[1, 1].set_title('Силуэт анализ')
axes[1, 1].set_xlabel('Количество кластеров')
axes[1, 1].set_ylabel('Силуэт коэффициент')

# 6. Распределение вероятностей (как в оригинале)
axes[1, 2].hist(gmm_max_probs, bins=30, alpha=0.7, color='skyblue')
axes[1, 2].axvline(x=uncertainty_threshold, color='red', linestyle='--',
                   label=f'Порог неопределенности: {uncertainty_threshold}')
axes[1, 2].set_title('GMM: Распределение вероятностей')
axes[1, 2].set_xlabel('Максимальная вероятность')
axes[1, 2].set_ylabel('Количество клиентов')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# === БИЗНЕС-АНАЛИЗ ===
print("\n💼 Шаг 5: Бизнес-анализ гибридных сегментов...")


def analyze_business_value(segment_data, segment_name):
    """Анализ бизнес-ценности сегмента"""
    size = len(segment_data)
    total_revenue = segment_data['total_amount'].sum()
    avg_revenue = segment_data['total_amount'].mean()
    avg_transactions = segment_data['transaction_count'].mean()

    # Потенциал роста
    growth_potential = (
        segment_data['unique_merchants'].mean() * 0.3 +
        segment_data['transaction_count'].mean() * 0.4 +
        (1 - segment_data['regularity_score'].mean()) * 0.3
    )

    # Риск-профиль
    risk_score = (
        segment_data['coefficient_variation'].mean() * 0.4 +
        segment_data['weekend_ratio'].mean() * 0.2 +
        segment_data['night_ratio'].mean() * 0.4
    )

    return {
        'segment': segment_name,
        'size': size,
        'total_revenue': total_revenue,
        'avg_revenue': avg_revenue,
        'avg_transactions': avg_transactions,
        'growth_potential': growth_potential,
        'risk_score': risk_score
    }


# Анализ всех сегментов
business_analysis = []
for segment in client_features['hybrid_segment'].unique():
    segment_data = client_features[client_features['hybrid_segment'] == segment]
    analysis = analyze_business_value(segment_data, segment)
    business_analysis.append(analysis)

business_df = pd.DataFrame(business_analysis)
business_df = business_df.sort_values('total_revenue', ascending=False)

print("🏆 Рейтинг сегментов по доходности:")
for _, row in business_df.head(10).iterrows():
    print(f"  {row['segment']}: {row['total_revenue']:,.0f} тенге "
          f"({row['size']:,} клиентов, {row['avg_revenue']:,.0f} ср.доход)")

# === БАНКОВСКИЕ РЕКОМЕНДАЦИИ ===
print(f"\n🎯 Банковские рекомендации по сегментам:")


def generate_banking_recommendations(segment_data, segment_name):
    """Генерация банковских рекомендаций для сегмента"""
    avg_amount = segment_data['total_amount'].mean()
    avg_transactions = segment_data['transaction_count'].mean()
    weekend_ratio = segment_data['weekend_ratio'].mean()
    unique_merchants = segment_data['unique_merchants'].mean()
    regularity = segment_data['regularity_score'].mean()
    gmm_certainty = segment_data['gmm_max_prob'].mean()

    recommendations = []
    segment_type = "Основной" if segment_name.startswith(
        'MAIN_') else "Уточненный"

    # Банковские продукты по доходности
    if avg_amount > 2000000:  # High-value
        recommendations.extend([
            "🏆 VIP-статус с персональным менеджером",
            "💎 Премиальные банковские карты (World Elite, Infinite)",
            "🏠 Ипотечные программы с льготными ставками",
            "📈 Инвестиционные продукты и частное банковское обслуживание"
        ])
    elif avg_amount > 800000:  # Medium-high value
        recommendations.extend([
            "💳 Премиальные карты с повышенным кэшбеком",
            "🎯 Целевые кредиты (авто, образование) с льготными условиями",
            "📊 Депозиты и накопительные продукты",
            "🛡️ Страховые продукты"
        ])
    elif avg_amount > 300000:  # Medium value
        recommendations.extend([
            "💰 Программы лояльности и кэшбек",
            "🏪 Рассрочка в партнерских магазинах",
            "📱 Мобильные платежи и цифровые сервисы"
        ])
    else:  # Mass market
        recommendations.extend([
            "📲 Базовые цифровые сервисы",
            "💡 Финансовая грамотность и обучающие программы",
            "🎁 Микро-бонусы за активное использование"
        ])

    # Дополнительные рекомендации
    if weekend_ratio > 0.4:
        recommendations.append("🎪 Weekend-маркетинг и специальные предложения")

    if unique_merchants < 5:
        recommendations.append(
            "🌐 Расширение торговой сети и новые категории партнеров")

    if regularity > 0.7:
        recommendations.append("🔄 Автоплатежи и подписочные сервисы")

    if segment_name.startswith('REFINED_'):
        recommendations.append(
            "📊 Детальная персонализация через машинное обучение")

    return {
        'segment': segment_name,
        'type': segment_type,
        'certainty': gmm_certainty,
        'recommendations': recommendations[:5]  # Топ-5 рекомендаций
    }


# Генерируем рекомендации для топ-сегментов
print("💡 Банковские рекомендации по сегментам:")
for _, row in business_df.head(8).iterrows():
    segment_name = row['segment']
    segment_data = client_features[client_features['hybrid_segment']
                                   == segment_name]
    rec_data = generate_banking_recommendations(segment_data, segment_name)

    print(f"\n🔹 {segment_name} ({row['size']:,} клиентов)")
    print(
        f"   Тип: {rec_data['type']} (уверенность GMM: {rec_data['certainty']:.3f})")
    print(f"   Доход: {row['total_revenue']:,.0f} тенге")
    for i, rec in enumerate(rec_data['recommendations'], 1):
        print(f"   {i}. {rec}")

# === СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
print(f"\n💾 Шаг 6: Сохранение результатов...")

# Основной файл с результатами (как в оригинале)
output_cols = ['card_id', 'total_amount', 'avg_amount', 'transaction_count',
               'unique_merchants', 'weekend_ratio', 'regularity_score',
               'gmm_cluster', 'gmm_max_prob', 'gmm_uncertain',
               'spectral_cluster', 'hybrid_segment']

# Добавляем вероятности GMM (как в оригинале)
for i in range(optimal_n_gmm):
    output_cols.append(f'prob_cluster_{i}')

final_results = client_features[output_cols].copy()
final_results.to_csv('hybrid_full_gmm_spectral_segments.csv', index=False)
print("✅ Гибридные сегменты сохранены в 'hybrid_full_gmm_spectral_segments.csv'")

# Бизнес-аналитика
business_df.to_csv('hybrid_full_business_analysis.csv', index=False)
print("✅ Бизнес-анализ сохранен в 'hybrid_full_business_analysis.csv'")

# Профили сегментов (как в оригинале)
segment_profiles = client_features.groupby('hybrid_segment')[feature_cols].agg([
    'mean', 'std', 'median']).round(3)
segment_profiles.to_csv('hybrid_full_segment_profiles.csv')
print("✅ Профили сегментов сохранены в 'hybrid_full_segment_profiles.csv'")

# Детальные профили GMM кластеров (как в оригинале)
gmm_cluster_profiles = client_features.groupby(
    'gmm_cluster')[feature_cols].agg(['mean', 'median', 'std']).round(3)
gmm_cluster_profiles.to_csv('gmm_cluster_profiles.csv')
print("✅ Профили GMM кластеров сохранены в 'gmm_cluster_profiles.csv'")

# Сводка модели (как в оригинале)
model_summary = {
    'approach': 'Hybrid Full GMM + Controlled SpectralClustering',
    'gmm_clusters': optimal_n_gmm,
    'gmm_covariance': best_cov_type,
    'gmm_bic_score': final_models[best_cov_type]['bic'],
    'gmm_aic_score': final_models[best_cov_type]['aic'],
    'gmm_log_likelihood': final_models[best_cov_type]['log_likelihood'],
    'gmm_silhouette_score': silhouette_scores[optimal_n_gmm - 2] if len(silhouette_scores) > optimal_n_gmm - 2 else 0,
    'spectral_clusters': n_spectral_clusters,
    'total_segments': len(client_features['hybrid_segment'].unique()),
    'main_segments': len(main_segments),
    'refined_segments': len(refined_segments) if len(refined_segments) > 0 else 0,
    'uncertain_clients': uncertain_clients,
    'uncertainty_percent': uncertain_clients / n_clients * 100,
    'total_clients': n_clients,
    'features_used': len(feature_cols)
}

pd.DataFrame([model_summary]).to_csv(
    'hybrid_full_model_summary.csv', index=False)
print("✅ Сводка модели сохранена в 'hybrid_full_model_summary.csv'")

# === ИТОГОВЫЙ ОТЧЕТ ===
print(f"\n🎉 ГИБРИДНЫЙ АНАЛИЗ: ПОЛНЫЙ GMM + КОНТРОЛИРУЕМЫЙ SPECTRAL ЗАВЕРШЕН!")
print("=" * 75)
print(f"📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"• Подход: Полный GMM + Контролируемый SpectralClustering")
print(f"• GMM кластеров: {optimal_n_gmm} (ковариация: {best_cov_type})")
print(f"• GMM BIC score: {final_models[best_cov_type]['bic']:.0f}")
print(f"• Spectral микрокластеров: {n_spectral_clusters}")
print(
    f"• Всего гибридных сегментов: {len(client_features['hybrid_segment'].unique())}")
print(f"• Основных сегментов: {len(main_segments)}")
print(
    f"• Уточненных сегментов: {len(refined_segments) if len(refined_segments) > 0 else 0}")
print(f"• Неопределенных клиентов: {uncertain_clients:,}")

print(f"\n🚀 ПРЕИМУЩЕСТВА ГИБРИДНОГО ПОДХОДА:")
print("• Полный GMM анализ с оптимальным выбором параметров")
print("• SpectralClustering только для уточнения неопределенных клиентов")
print("• Максимальное качество основных сегментов")
print("• Контролируемое количество дополнительных сегментов")
print("• Детальные профили и рекомендации для каждого сегмента")

print(f"\n💼 БИЗНЕС-ЦЕННОСТЬ:")
top_segment = business_df.iloc[0]
print(f"• Самый доходный сегмент: {top_segment['segment']}")
print(f"• Доход топ-сегмента: {top_segment['total_revenue']:,.0f} тенге")
print(f"• Средний доход клиента: {top_segment['avg_revenue']:,.0f} тенге")
print(
    f"• GMM качество: BIC={final_models[best_cov_type]['bic']:.0f}, силуэт={silhouette_scores[optimal_n_gmm - 2] if len(silhouette_scores) > optimal_n_gmm - 2 else 'N/A'}")

if len(refined_segments) > 0:
    refined_revenue = client_features[client_features['hybrid_segment'].str.startswith(
        'REFINED_')]['total_amount'].sum()
    print(f"• Доход от уточненных сегментов: {refined_revenue:,.0f} тенге")

print(f"\n🎯 ГОТОВО ДЛЯ БАНКОВСКОГО ВНЕДРЕНИЯ:")
print("• Персонализированные банковские продукты по сегментам")
print("• Таргетированный маркетинг для каждой группы клиентов")
print("• VIP-программы для высокодоходных сегментов")
print("• Детальная персонализация для уточненных сегментов")
print("• Полный набор файлов для анализа и внедрения")

print(f"\n✨ Лучшее из двух миров: качество GMM + точность SpectralClustering!")
print("=" * 75)

# Освобождение памяти
gc.collect()
