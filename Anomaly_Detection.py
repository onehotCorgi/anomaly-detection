# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from pycaret.anomaly import *

# %%
np.random.seed(42)  # 再現性のため固定

today = pd.Timestamp.today().normalize()  # 今日の 00:00
week_ago = today - pd.Timedelta(days=7)
three_days_ago = today - pd.Timedelta(days=3)
print(f'today:{today}')

n_samples = 7*6*24 # 10分ごと7日間のデータ
n_features = 2
n_anomalies = int(0.001 * n_samples)  # 挿入する異常点の数

# --- 正常な多変量正規分布データ ---
data = np.random.normal(loc=100, scale=1, size=(n_samples, n_features))

# --- 異常点をランダムに挿入 ---
anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)

# 各異常点にランダムな異常値を加算または減算
for idx in anomaly_indices:
    direction = np.random.choice([1, -1])
    magnitude = np.random.uniform(1, 3, size=n_features) #任意設定
    data[idx] += direction * magnitude

# --- DataFrame に変換 ---
time_index = pd.date_range(start=week_ago, periods=n_samples, freq='10T') # 10分間隔
df = pd.DataFrame(data, columns=['sensor_1', 'sensor_2'])
df['timestamp'] = time_index

# 異常ラベルを追加
df['is_anomaly'] = 0
df.loc[anomaly_indices, 'is_anomaly'] = 1

print(df.head())

# %%
# フィルタリング
df_original = df[(df['timestamp'] >= three_days_ago) & (df['timestamp'] <= today)]
df_ans = df_original.copy()  # df_ans retains all columns, including 'is_anomaly'
print(f'\ndf_ans:\n{df_ans.head()}\n')
if 'is_anomaly' in df_ans.columns:
	print('anomaly:')
	print(df_ans[df_ans['is_anomaly'] == 1])
else:
	print("'is_anomaly' column not found in df_ans")
df = df_original[['sensor_1', 'sensor_2', 'timestamp']]
print(f'\ndf:\n{df.head()}\n')

# %%
# --- 2. PyCaret セットアップ ---
exp_anomaly_detection = setup(
    data=df,
    session_id=123,
    verbose=False
    ) 


# %%
# 2. モデルの作成と比較
available_models = models()
models_to_use = available_models.index.tolist()
results_dict = {}
results_list = []

for model_id in models_to_use:
    print(f"Training {model_id.upper()}...")
    model = create_model(model_id)
    assigned = assign_model(model)
    results_dict[model_id] = assigned

# 3. 可視化：時系列プロットに異常マーカーを表示
fig, axs = plt.subplots(len(models_to_use), 1, figsize=(12, 4*len(models_to_use)), sharex=True)

for i, model_id in enumerate(models_to_use):
    result = results_dict[model_id]
    y_true = df_ans['is_anomaly']
    y_pred = result['Anomaly']  # PyCaret の予測結果
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    results_list.append({
        'Model': model_id,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
    axs[i].plot(result['timestamp'], result['sensor_1'], label='sensor_1')
    axs[i].scatter(result['timestamp'][result['Anomaly'] == 1],
                   result['sensor_1'][result['Anomaly'] == 1],
                   color='red', label='Anomaly', s=50, marker='x')
    axs[i].set_title(f"{model_id.upper()} - sensor_1 Anomalies")
    axs[i].legend()
    
plt.tight_layout()
plt.show()

df_scores = pd.DataFrame(results_list)
print(f'df_scores:\n{df_scores}')

df_melted = df_scores.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric')
plt.title("Model Evaluation on Injected Anomalies")
plt.ylim(0, 1.05)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()


# %%
#model_ids = df_scores[df_scores['Precision']>= 0.05].loc[:, 'Model']
model_ids = df_scores[df_scores['Recall']>= 0.7].loc[:, 'Model'] # 任意設定
model_ids


# %%
# センサー1とセンサー2の平均と標準偏差を計算
mean = df[['sensor_1', 'sensor_2']].mean()
std = df[['sensor_1', 'sensor_2']].std()

# 各点が 3σ の範囲外かどうかを判定
anomaly_mask = (
    (df['sensor_1'] < mean['sensor_1'] - 3 * std['sensor_1']) |
    (df['sensor_1'] > mean['sensor_1'] + 3 * std['sensor_1']) |
    (df['sensor_2'] < mean['sensor_2'] - 3 * std['sensor_2']) |
    (df['sensor_2'] > mean['sensor_2'] + 3 * std['sensor_2'])
)

# 異常ラベル列を作成（すでに 'is_anomaly' がある場合は上書き）
df['is_anomaly_3sigma'] = anomaly_mask.astype(int)
print(df['is_anomaly_3sigma'].value_counts())

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['timestamp'], df['sensor_1'], label='Sensor 1', alpha=0.7)
ax.plot(df['timestamp'], df['sensor_2'], label='Sensor 2', alpha=0.7)

# 異常点をオレンジでマーク
anomalies = df[df['is_anomaly_3sigma'] == 1]
ax.scatter(anomalies['timestamp'], anomalies['sensor_1'], color='orange', label='Anomaly (3sigma)', s=50, marker='x')
ax.scatter(anomalies['timestamp'], anomalies['sensor_2'], color='black', s=50, marker='x')

ax.set_title('Anomalies Detected by 3sigma')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
# --- 3. すべての異常検知モデルを作成 ---
results_dict = {}
for model_id in model_ids:
    try:
        print(f"Training {model_id}...")
        model = create_model(model_id)
        results_dict[model_id] = assign_model(model)
    except Exception as e:
        print(f"Skipped {model_id}: {e}")

# --- 4. アンサンブル処理 ---
# スコアアンサンブル用のDataFrame
score_df = pd.DataFrame(index=df.index)
vote_df = pd.DataFrame(index=df.index)

# 正規化スコア＋フラグの統合
for model_id, result_df in results_dict.items():
    # スコアアンサンブル
    if 'Anomaly_Score' in result_df.columns:
        # Check if all values are finite
        scores = result_df['Anomaly_Score'].values
        if np.all(np.isfinite(scores)):
            scaler = MinMaxScaler()
            normalized_score = scaler.fit_transform(result_df[['Anomaly_Score']])
            score_df[model_id] = normalized_score.flatten()
        else:
            print(f"Skipped {model_id} for score ensemble due to non-finite values in Anomaly_Score.")

    # 投票アンサンブル
    if 'Anomaly' in result_df.columns:
        vote_df[model_id] = result_df['Anomaly']

# スコア平均
score_df['ensemble_score'] = score_df.mean(axis=1)

# 投票
if vote_df.empty:
    print("No models available for voting ensemble.")
    vote_df['ensemble_vote'] = 0  # 投票アンサンブルができない場合は全て0に設定
else:
    vote_ratio = 1 # 投票したモデル割合
    vote_df['ensemble_vote'] = (vote_df.sum(axis=1) >= vote_df.shape[1] * vote_ratio).astype(int)

# スコアの上位xx%を異常と判定（スコアアンサンブル）
threshold = score_df['ensemble_score'].quantile(0.99)
score_df['ensemble_flag'] = (score_df['ensemble_score'] >= threshold).astype(int)

# --- 5. 結果統合 ---
df_result = df.copy()
df_result['score_ensemble'] = score_df['ensemble_score']
df_result['score_flag'] = score_df['ensemble_flag']
df_result['vote_flag'] = vote_df['ensemble_vote']

# --- 6. 可視化 ---
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# 元データ
axs[0].plot(df_result['timestamp'], df_result['sensor_1'], label='sensor_1')
axs[0].set_title("Original Sensor Data")

# スコアアンサンブルでの異常
axs[1].plot(df_result['timestamp'], df_result['sensor_1'])
axs[1].scatter(df_result['timestamp'][df_result['score_flag'] == 1],
               df_result['sensor_1'][df_result['score_flag'] == 1],
               color='red', label='Score Anomaly', s=50, marker='x')
axs[1].set_title("Anomalies (Score Ensemble)")

# 投票アンサンブルでの異常
axs[2].plot(df_result['timestamp'], df_result['sensor_1'])
axs[2].scatter(df_result['timestamp'][df_result['vote_flag'] == 1],
               df_result['sensor_1'][df_result['vote_flag'] == 1],
               color='purple', label='Vote Anomaly', s=50, marker='x')
axs[2].set_title("Anomalies (Voting Ensemble)")

for ax in axs:
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()


# %%
# 可視化関数（ラベルごとに表示）
def plot_anomaly_scatter(data, label_col, title, color):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x='sensor_1',
        y='sensor_2',
        hue=label_col,
        palette={0: 'gray', 1: color},
        alpha=0.6
    )
    plt.title(f"2D Anomaly Detection ({title})", fontsize=14)
    plt.xlabel("Sensor 1")
    plt.ylabel("Sensor 2")
    plt.legend(title="Anomaly", loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 実行
plot_anomaly_scatter(df, 'is_anomaly_3sigma', '3sigma Rule', 'orange')
plot_anomaly_scatter(df_result, 'score_flag', 'Score Ensemble', 'red')
plot_anomaly_scatter(df_result, 'vote_flag', 'Voting Ensemble', 'purple')
plot_anomaly_scatter(df_ans, 'is_anomaly', 'Answer', 'green')



