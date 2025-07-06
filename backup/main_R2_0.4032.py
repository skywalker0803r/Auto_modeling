import pandas as pd

# loading
df = pd.read_excel('20240916-CFB2脫硫劑優化改善.xlsx')
col = df.columns
df = df.iloc[1:,:]
df.columns = col
df = df.set_index('Unnamed: 0')
df.index.name = 'datetime'

# to numeric
for i in df.columns:
  df[i] = pd.to_numeric(df[i], errors='coerce')

# select by rule
# rule1
coal_low = df[df["MLUT4_FIQ-2BTCF"] < 20]
# rule2
sox = df["MLUT4_AT-240"]
constant_sox_indices = sox[sox.shift(1) == sox][(sox.shift(2) == sox) & (sox.shift(-1) == sox)].index
constant_sox = df.loc[constant_sox_indices]
# rule1 union rule2
common_index = coal_low.index.union(constant_sox.index)
# select by rule1 union rule2
select_df = df.loc[~df.index.isin(common_index), :]

# features
import pickle
with open("features1.pkl", "rb") as f:
    features = pickle.load(f)

# target
y_col = 'DeSOx_1st'

# select features and target
select_df = select_df[features+[y_col]]
select_df

# start modeling
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def train_model(train_X, train_y):
    model = RandomForestRegressor(n_estimators=2000, random_state=42, n_jobs=-1, min_samples_leaf=1, max_depth=None, max_features='sqrt', criterion='absolute_error')
    return model.fit(train_X, train_y)

# 1. 時間排序
select_df = select_df.sort_index().reset_index(drop=True)

# 2. 設定
target_col = "DeSOx_1st"
feature_cols = [col for col in select_df.columns if col not in ['timestamp', target_col]]
start_idx = 1000
total_steps = len(select_df) - start_idx
total_steps = len(select_df) - start_idx

# 3. 儲存預測結果
predictions = []
abs_errors = []
thresholds = []
indices = []

# 4. 初始化進度條
pbar = tqdm(total=total_steps)

# 5. 動態預測迴圈
i = start_idx
while i < len(select_df):
    # 早停判斷<不得移除>
    if pbar.n >= 100:
        break

    # 判斷是否要建模 並記錄
    if 'current_model' not in locals() or error_exceeded_threshold:
        train_df = select_df.iloc[i - 1000:i]
        train_X = train_df[feature_cols]
        train_y = train_df[target_col]

        # train model start
        current_model = train_model(train_X, train_y)

        # Calculate 80th percentile error threshold for the newly trained model
        train_preds = current_model.predict(train_X)
        train_errors = np.abs(train_preds - train_y)
        current_threshold = np.percentile(train_errors, 90)
        error_exceeded_threshold = False # Reset flag

    # 預測i
    test_row = select_df.iloc[i]
    test_X = test_row[feature_cols].values.reshape(1, -1)
    true_y = test_row[target_col]
    pred_y = current_model.predict(test_X)[0]
    error = abs(pred_y - true_y)

    # 紀錄
    predictions.append(pred_y)
    abs_errors.append(error)
    thresholds.append(current_threshold)
    indices.append(i)

    # 更新進度條
    pbar.update(1)

    # 根據預測i的結果判斷是否要設置error_exceeded_threshold
    if error > current_threshold:
        # If error exceeds, a new model will be trained in the next iteration
        # and 'i' increments by 1
        i += 1
        error_exceeded_threshold = True
    else:
        # If error does not exceed, stay with the current model for the next prediction
        # 'i' increments by 1, and the loop will try to predict 'i+1' with the *same* model
        i += 1
        error_exceeded_threshold = False # Continue using current model
# 結束訓練
pbar.close()

# 6. 輸出結果表
result_df = select_df.loc[indices].copy()
result_df['prediction'] = predictions
result_df['abs_error'] = abs_errors
result_df['threshold'] = thresholds

# 預覽結果
print(result_df.head())

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# 指標評估
y_true = result_df['DeSOx_1st']
y_pred = result_df['prediction']
# Calculate R-squared
r2 = r2_score(y_true, y_pred)
# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# Calculate MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100 if np.any(y_true != 0) else 0

print(f"R-squared: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")




    