# Gaussian Mixture Models для банковской кластеризации клиентов
# Автор: Erik (Decentra) - Оптимальное решение для финансовых данных

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
import warnings
import os
import gc
warnings.filterwarnings('ignore')

plt.style.use('default')

print("🎯 GAUSSIAN MIXTURE MODELS ДЛЯ БАНКОВСКИХ ДАННЫХ")
print("=" * 55)
print("🔍 Цель: Мягкая кластеризация с вероятностями принадлежности")
print("⚡ Преимущества: Автоматический выбор кластеров + качество для финансов")

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("\n📊 Шаг 1: Загрузка и подготовка данных...")

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

        # Для GMM оптимальная выборка - больше чем для HDBSCAN
        MAX_CLIENTS_GMM = 30000
        unique_clients_total = df['card_id'].nunique()

        if unique_clients_total > MAX_CLIENTS_GMM:
            print(
                f"⚠️ Много клиентов ({unique_clients_total:,}), создаем выборку: {MAX_CLIENTS_GMM:,}")

            # Качественная стратифицированная выборка
            client_summary = df.groupby('card_id').agg({
                'transaction_amount_kzt': ['count', 'sum', 'mean', 'std'],
                'transaction_timestamp': ['min', 'max'],
                'merchant_id': 'nunique',
                'mcc_category': 'nunique'
            })

            client_summary.columns = ['txn_count', 'total_amount', 'avg_amount', 'std_amount',
                                      'first_txn', 'last_txn', 'unique_merchants', 'unique_categories']

            # Многомерная стратификация
            client_summary['activity_level'] = pd.qcut(
                client_summary['txn_count'], q=5, labels=range(5))
            client_summary['amount_level'] = pd.qcut(
                client_summary['total_amount'], q=4, labels=range(4))
            client_summary['diversity_level'] = pd.qcut(
                client_summary['unique_merchants'], q=3, labels=range(3))

            # Равномерная выборка из каждой страты
            sample_clients = []
            clients_per_strata = MAX_CLIENTS_GMM // 60  # 5*4*3 = 60 страт

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

# 2. ОЧИСТКА И ОБОГАЩЕНИЕ ДАННЫХ
print("\n🧹 Шаг 2: Очистка и обогащение данных...")

original_size = len(df)
df = df[df['transaction_amount_kzt'] > 0]
df['transaction_timestamp'] = pd.to_datetime(
    df['transaction_timestamp'], errors='coerce')
df = df[df['transaction_timestamp'].notna()]

# Добавляем временные и поведенческие признаки
df['hour'] = df['transaction_timestamp'].dt.hour
df['day_of_week'] = df['transaction_timestamp'].dt.dayofweek
df['month'] = df['transaction_timestamp'].dt.month
df['quarter'] = df['transaction_timestamp'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

# Категории времени
df['time_category'] = pd.cut(df['hour'],
                             bins=[0, 6, 12, 18, 24],
                             labels=['Night', 'Morning', 'Day', 'Evening'],
                             include_lowest=True)

print(f"Очистка: {original_size:,} → {len(df):,} записей")

# 3. СОЗДАНИЕ ПРИЗНАКОВ ДЛЯ GMM
print("\n🔧 Шаг 3: Создание признаков для GMM...")

# Базовые финансовые метрики
print("  💰 Финансовые метрики...")
financial_features = df.groupby('card_id').agg({
    'transaction_amount_kzt': ['sum', 'mean', 'median', 'std', 'count', 'min', 'max'],
    'transaction_timestamp': ['min', 'max']
}).reset_index()

financial_features.columns = ['card_id', 'total_amount', 'avg_amount', 'median_amount',
                              'std_amount', 'transaction_count', 'min_amount', 'max_amount',
                              'first_transaction', 'last_transaction']

# Поведенческие метрики
print("  🛍️ Поведенческие метрики...")
behavioral_features = df.groupby('card_id').agg({
    'merchant_id': 'nunique',
    'merchant_city': 'nunique',
    'mcc_category': 'nunique',
    'transaction_type': 'nunique'
}).reset_index()

behavioral_features.columns = ['card_id', 'unique_merchants', 'unique_cities',
                               'unique_categories', 'unique_txn_types']

# Временные паттерны
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

# Топ категории MCC (для GMM берем больше)
print("  🏪 Категории покупок...")
top_categories = df['mcc_category'].value_counts().head(12).index.tolist()
mcc_features = df.groupby('card_id')['mcc_category'].apply(
    lambda x: pd.Series({f'mcc_{cat.lower()}_ratio': (
        x == cat).mean() for cat in top_categories})
).reset_index()

# Дополнительные GMM-специфичные признаки
print("  📊 Продвинутые метрики...")


def calculate_gmm_features(group):
    amounts = group['transaction_amount_kzt']
    timestamps = group['transaction_timestamp']

    # Временные интервалы
    time_diffs = timestamps.diff().dt.total_seconds() / 3600
    time_diffs = time_diffs.dropna()

    # Статистики распределения
    return pd.Series({
        'amount_skewness': amounts.skew(),
        'amount_kurtosis': amounts.kurtosis(),
        'amount_range_ratio': (amounts.max() - amounts.min()) / amounts.mean() if amounts.mean() > 0 else 0,
        'large_txn_ratio': (amounts > amounts.quantile(0.9)).mean(),
        'small_txn_ratio': (amounts < amounts.quantile(0.1)).mean(),
        'regularity_score': 1 / (1 + amounts.std() / amounts.mean()) if amounts.mean() > 0 else 0,
        'avg_time_between_txns': time_diffs.mean() if len(time_diffs) > 0 else 0,
        'time_consistency': 1 / (1 + time_diffs.std() / time_diffs.mean()) if len(time_diffs) > 0 and time_diffs.mean() > 0 else 0
    })


advanced_features = df.groupby('card_id').apply(
    calculate_gmm_features).reset_index()

# Объединяем все признаки
print("  🔗 Объединение признаков...")
client_features = financial_features.merge(
    behavioral_features, on='card_id', how='left')
client_features = client_features.merge(
    time_features, on='card_id', how='left')
client_features = client_features.merge(mcc_features, on='card_id', how='left')
client_features = client_features.merge(
    advanced_features, on='card_id', how='left')

# Вычисляемые признаки
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

# Проверка и удаление дубликатов
print("🔧 Проверка дубликатов...")
duplicates = client_features['card_id'].duplicated().sum()
if duplicates > 0:
    print(f"⚠️ Найдено {duplicates} дубликатов, удаляем...")
    client_features = client_features.drop_duplicates(
        subset=['card_id'], keep='first')

# Финальная обработка
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

# 4. ПОДГОТОВКА ДАННЫХ ДЛЯ GMM
print("\n⚙️ Шаг 4: Подготовка данных для GMM...")

# Выбираем признаки для GMM (все числовые кроме ID)
gmm_features = client_features.select_dtypes(
    include=[np.number]).columns.tolist()
gmm_features.remove('card_id')

print(f"Признаков для GMM: {len(gmm_features)}")

# Подготовка матрицы
X = client_features[gmm_features].copy()

# Мягкая обработка выбросов (GMM более устойчив к выбросам)
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

# 5. ОПТИМИЗАЦИЯ GAUSSIAN MIXTURE MODEL
print("\n🎯 Шаг 5: Оптимизация GMM...")

n_clients = len(X_scaled)
print(f"Клиентов для анализа: {n_clients:,}")

# Тестируем разное количество компонент
print("📊 Поиск оптимального количества компонент...")

n_components_range = range(2, 16)  # Тестируем от 2 до 15 кластеров
bic_scores = []
aic_scores = []
silhouette_scores = []
log_likelihood_scores = []

for n_comp in n_components_range:
    print(f"  Тестируем {n_comp} компонент...", end="")

    # Тестируем разные типы ковариационных матриц
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

        # Метрики качества
        bic = best_model.bic(X_scaled)
        aic = best_model.aic(X_scaled)
        log_likelihood = best_model.score(X_scaled)

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
        log_likelihood_scores.append(log_likelihood)

        print(f" BIC: {bic:.0f}, AIC: {aic:.0f}, Силуэт: {sil_score:.3f}")
    else:
        print(" Ошибка")
        bic_scores.append(np.inf)
        aic_scores.append(np.inf)
        silhouette_scores.append(-1)
        log_likelihood_scores.append(-np.inf)

# Выбор оптимального количества компонент
optimal_n_bic = n_components_range[np.argmin(bic_scores)]
optimal_n_aic = n_components_range[np.argmin(aic_scores)]
optimal_n_sil = n_components_range[np.argmax(silhouette_scores)]

print(f"\n📊 Рекомендации по количеству кластеров:")
print(f"• По BIC: {optimal_n_bic} кластеров")
print(f"• По AIC: {optimal_n_aic} кластеров")
print(f"• По силуэту: {optimal_n_sil} кластеров")

# Выбираем компромиссное решение
optimal_n = optimal_n_bic  # BIC обычно лучше для выбора модели
print(f"✅ Выбираем: {optimal_n} кластеров (по BIC)")

# 6. ФИНАЛЬНАЯ GMM МОДЕЛЬ
print(f"\n🎯 Шаг 6: Обучение финальной GMM модели...")

# Тестируем разные типы ковариационных матриц для финальной модели
final_models = {}
for cov_type in ['full', 'tied', 'diag']:
    try:
        gmm = GaussianMixture(
            n_components=optimal_n,
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

# Выбираем лучшую модель
best_cov_type = min(final_models.keys(), key=lambda x: final_models[x]['bic'])
final_gmm = final_models[best_cov_type]['model']

print(f"✅ Лучшая модель: {best_cov_type} covariance")

# Получаем результаты
final_labels = final_gmm.predict(X_scaled)
probabilities = final_gmm.predict_proba(X_scaled)
log_likelihood = final_gmm.score(X_scaled)

# Анализ неопределенности
max_probs = probabilities.max(axis=1)
uncertainty_threshold = 0.6
uncertain_clients = (max_probs < uncertainty_threshold).sum()

print(f"\n📊 Результаты GMM:")
print(f"• Кластеров: {optimal_n}")
print(f"• Тип ковариации: {best_cov_type}")
print(f"• Log-likelihood: {log_likelihood:.2f}")
print(
    f"• Неопределенных клиентов: {uncertain_clients} ({uncertain_clients/n_clients*100:.1f}%)")

# Добавляем результаты в датафрейм
client_features['gmm_cluster'] = final_labels
client_features['max_probability'] = max_probs
client_features['is_uncertain'] = (max_probs < uncertainty_threshold)

# Добавляем вероятности для каждого кластера
for i in range(optimal_n):
    client_features[f'prob_cluster_{i}'] = probabilities[:, i]

# 7. АНАЛИЗ КЛАСТЕРОВ
print(f"\n📈 Шаг 7: Анализ GMM кластеров...")

print("📊 Распределение клиентов по кластерам:")
cluster_sizes = pd.Series(final_labels).value_counts().sort_index()
for cluster_id, size in cluster_sizes.items():
    percentage = size / n_clients * 100
    avg_prob = probabilities[final_labels == cluster_id, cluster_id].mean()
    print(
        f"  Кластер {cluster_id}: {size:,} клиентов ({percentage:.1f}%), ср.вероятность: {avg_prob:.3f}")

# Профили кластеров
print(f"\n💡 Профили GMM кластеров:")
key_metrics = ['total_amount', 'avg_amount', 'transaction_count', 'unique_merchants',
               'weekend_ratio', 'business_hours_ratio', 'regularity_score']

for cluster_id in sorted(pd.Series(final_labels).unique()):
    cluster_data = client_features[client_features['gmm_cluster'] == cluster_id]
    size = len(cluster_data)
    avg_certainty = cluster_data['max_probability'].mean()

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

# Анализ неопределенных клиентов
if uncertain_clients > 0:
    uncertain_data = client_features[client_features['is_uncertain']]
    print(
        f"\n🔍 Анализ неопределенных клиентов ({uncertain_clients} клиентов):")
    print(
        f"  • Средняя общая сумма: {uncertain_data['total_amount'].mean():,.0f} тенге")
    print(f"  • Средний чек: {uncertain_data['avg_amount'].mean():,.0f} тенге")
    print(
        f"  • Среднее кол-во транзакций: {uncertain_data['transaction_count'].mean():.0f}")
    print(
        f"  • Средняя максимальная вероятность: {uncertain_data['max_probability'].mean():.3f}")

# 8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("\n🎨 Шаг 8: Визуализация GMM результатов...")

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Создание графиков
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Gaussian Mixture Model: {optimal_n} кластеров', fontsize=16)

# 1. Основная визуализация кластеров
colors = plt.cm.tab10(np.linspace(0, 1, optimal_n))
for i in range(optimal_n):
    mask = final_labels == i
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[colors[i]], label=f'Кластер {i}', alpha=0.7, s=30)

axes[0, 0].set_title('GMM кластеры (PCA проекция)')
axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0, 0].legend()

# 2. Карта неопределенности
scatter = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=max_probs,
                             cmap='viridis', alpha=0.6, s=20)
axes[0, 1].set_title('Карта уверенности')
axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.colorbar(scatter, ax=axes[0, 1])

# 3. Размеры кластеров
axes[0, 2].bar(range(optimal_n), cluster_sizes.values, color=colors)
axes[0, 2].set_title('Размеры кластеров')
axes[0, 2].set_xlabel('Кластер')
axes[0, 2].set_ylabel('Количество клиентов')

# 4. BIC/AIC кривые
axes[1, 0].plot(n_components_range, bic_scores, 'b-o', label='BIC')
axes[1, 0].plot(n_components_range, aic_scores, 'r-s', label='AIC')
axes[1, 0].axvline(x=optimal_n, color='green',
                   linestyle='--', label=f'Выбрано: {optimal_n}')
axes[1, 0].set_title('Критерии выбора модели')
axes[1, 0].set_xlabel('Количество кластеров')
axes[1, 0].set_ylabel('Значение критерия')
axes[1, 0].legend()

# 5. Силуэт анализ
axes[1, 1].plot(n_components_range, silhouette_scores, 'g-^')
axes[1, 1].axvline(x=optimal_n, color='green', linestyle='--')
axes[1, 1].set_title('Силуэт анализ')
axes[1, 1].set_xlabel('Количество кластеров')
axes[1, 1].set_ylabel('Силуэт коэффициент')

# 6. Распределение вероятностей
axes[1, 2].hist(max_probs, bins=30, alpha=0.7, color='skyblue')
axes[1, 2].axvline(x=uncertainty_threshold, color='red', linestyle='--',
                   label=f'Порог неопределенности: {uncertainty_threshold}')
axes[1, 2].set_title('Распределение максимальных вероятностей')
axes[1, 2].set_xlabel('Максимальная вероятность')
axes[1, 2].set_ylabel('Количество клиентов')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\n💾 Шаг 9: Сохранение результатов...")

# Основной файл с результатами
output_cols = ['card_id', 'total_amount', 'avg_amount', 'transaction_count',
               'activity_days', 'unique_merchants', 'unique_cities',
               'weekend_ratio', 'business_hours_ratio', 'regularity_score',
               'gmm_cluster', 'max_probability', 'is_uncertain']

# Добавляем столбцы с вероятностями
for i in range(optimal_n):
    output_cols.append(f'prob_cluster_{i}')

available_output_cols = [
    col for col in output_cols if col in client_features.columns]
final_results = client_features[available_output_cols].copy()

# Проверка на дубликаты
assert final_results['card_id'].nunique() == len(
    final_results), "Найдены дубликаты!"

final_results.to_csv('gmm_client_segments.csv', index=False)
print("✅ Результаты сохранены в 'gmm_client_segments.csv'")

# Детальные профили кластеров
cluster_profiles = client_features.groupby('gmm_cluster')[gmm_features].agg([
    'mean', 'median', 'std']).round(3)
cluster_profiles.to_csv('gmm_cluster_profiles.csv')
print("✅ Профили кластеров сохранены в 'gmm_cluster_profiles.csv'")

# Сводка модели
model_summary = {
    'algorithm': 'Gaussian Mixture Model',
    'n_clusters': optimal_n,
    'covariance_type': best_cov_type,
    'bic_score': final_models[best_cov_type]['bic'],
    'aic_score': final_models[best_cov_type]['aic'],
    'log_likelihood': log_likelihood,
    # -2 because range starts from 2
    'silhouette_score': silhouette_scores[optimal_n - 2],
    'total_clients': n_clients,
    'uncertain_clients': uncertain_clients,
    'uncertainty_percent': uncertain_clients / n_clients * 100,
    'features_used': len(gmm_features)
}

model_summary_df = pd.DataFrame([model_summary])
model_summary_df.to_csv('gmm_model_summary.csv', index=False)
print("✅ Сводка модели сохранена в 'gmm_model_summary.csv'")

print(f"\n🎉 GMM АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 50)
print(f"📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"• Алгоритм: Gaussian Mixture Model")
print(f"• Найдено кластеров: {optimal_n}")
print(f"• Тип ковариации: {best_cov_type}")
print(f"• BIC score: {final_models[best_cov_type]['bic']:.0f}")
