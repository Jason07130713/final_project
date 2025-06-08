import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('tesla.csv')

df.dropna(subset=['price'], inplace=True) # 如果 price 是空的，移除該列
df = df[pd.to_numeric(df['price'], errors='coerce').notnull()]
df['price'] = df['price'].astype(float)

for col in ['age', 'mile']:
    df = df[pd.to_numeric(df[col], errors='coerce').notnull()]
    df[col] = df[col].astype(float)


df = pd.get_dummies(df, columns=['model'], prefix='Model', drop_first=True) 



y = df['price']


X = df.drop('price', axis=1) 
if 'year' in X.columns: 
    X = X.drop('year', axis=1)


print("特徵 (X) 的欄位：")
print(X.columns)
print("特徵 (X) 前五筆：")
print(X.head())



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# --- (可選) 特徵縮放 ---
# 對於線性模型或梯度提升等模型，特徵縮放有時能改善表現
# 隨機森林對特徵縮放不敏感
scaler = StandardScaler()
# 只對訓練集進行 fit_transform，避免資料洩漏
X_train_scaled = scaler.fit_transform(X_train)
# 對測試集只進行 transform
X_test_scaled = scaler.transform(X_test)


lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)




y_pred_lr = lr_model.predict(X_test_scaled) 
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n線性迴歸模型評估結果：")
print(f"  平均絕對誤差 (MAE): {mae_lr:.2f} 萬元")
print(f"  均方根誤差 (RMSE): {rmse_lr:.2f} 萬元")
print(f"  R 平方分數 (R-squared): {r2_lr:.2f}")

print("\n\n--- Tesla 二手車價格預測 ---")

# 記住訓練時 X 的欄位順序和名稱 (在 train_test_split 之前的 X)
# 這是為了確保新輸入的資料有著完全相同的欄位結構
X_input_columns = X.columns.tolist() # X 是您在訓練前定義的特徵集 DataFrame
current_year_for_age_calc = 2025 # 確保這個年份與您 CSV 中 age 欄位的計算基準一致

try:
    input_model_name_raw = input("請輸入車型 (例如 Model 3, Model S, Model X, Model Y): ").strip()
    input_year_val = int(input("請輸入出廠年份 (例如 2022): "))
    input_mileage_val = float(input("請輸入行駛里程 (公里，例如 30000): "))
except ValueError:
    print("輸入錯誤，年份和里程必須是數字。請重新執行。")
    exit()

# 1. 預處理新輸入的數據
# 計算車齡
new_car_age = current_year_for_age_calc - input_year_val

# 創建一個與訓練時 X 特徵結構完全一致的 DataFrame
# X_input_columns 包含了所有獨熱編碼後的 'Model_xxx' 欄位以及 'age', 'mile'
new_car_input_df = pd.DataFrame(columns=X_input_columns)
new_car_input_df.loc[0] = 0 # 初始化所有特徵值為 0 (非常重要)

# 填入 age 和 mile
# 確保 'age' 和 'mile' 欄位確實存在於 X_input_columns 中
if 'age' in new_car_input_df.columns:
    new_car_input_df.loc[0, 'age'] = new_car_age
else:
    print("錯誤： 'age' 欄位不在模型訓練的特徵中，請檢查您的 CSV 檔案和預處理步驟。")
    exit()

if 'mile' in new_car_input_df.columns:
    new_car_input_df.loc[0, 'mile'] = input_mileage_val
else:
    print("錯誤： 'mile' 欄位不在模型訓練的特徵中，請檢查您的 CSV 檔案和預處理步驟。")
    exit()

# 處理獨熱編碼的車型特徵
model_ohe_column_to_set = f"Model_{input_model_name_raw}" # 例如 "Model_Model Y"

if model_ohe_column_to_set in X_input_columns: # 檢查這個獨熱編碼後的欄位是否存在於訓練特徵中
    new_car_input_df.loc[0, model_ohe_column_to_set] = 1
    print(f"設定特徵 '{model_ohe_column_to_set}' = 1")
else:
    # 如果欄位不存在，表示這個車型可能是 drop_first=True 時的基準車型
    # （例如，如果 Model 3 是基準，那麼所有 Model_xxx 欄位此時都應為0）
    # 或者是一個模型從未見過的車型
    print(f"輸入車型 '{input_model_name_raw}' (對應獨熱編碼欄位 '{model_ohe_column_to_set}') 被視為基準車型或訓練時未包含此明確欄位。")

print("\n準備好的預測輸入特徵 (獨熱編碼後，縮放前)：")
# 為了與 scaler 的輸入保持一致，欄位順序很重要
print(new_car_input_df[X_input_columns])


# 2. 使用訓練階段的 scaler 對新數據進行特徵縮放
# scaler 是在 X_train 上 fit 過的，所以這裡只做 transform
# 傳遞給 scaler.transform 的 DataFrame 欄位順序需要和 X_train 一致
try:
    new_car_input_scaled = scaler.transform(new_car_input_df[X_input_columns])
except ValueError as ve:
    print(f"\n對新數據進行特徵縮放時發生錯誤: {ve}")
    print(f"預期特徵數量 (來自 scaler.n_features_in_): {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'N/A'}")
    print(f"實際提供給 scaler 的特徵數量: {new_car_input_df[X_input_columns].shape[1]}")
    print(f"實際提供給 scaler 的特徵欄位: {new_car_input_df[X_input_columns].columns.tolist()}")
    print("請確保 'scaler' 物件已從訓練階段正確初始化，且輸入特徵的欄位與順序和訓練時完全一致。")
    exit()
except Exception as e:
    print(f"\n對新數據進行特徵縮放時發生未知錯誤: {e}")
    exit()


# 3. 使用訓練好的線性迴歸模型進行預測
try:
    predicted_price = lr_model.predict(new_car_input_scaled)
    print("\n==============================================")
    print(f"對於 車型: {input_model_name_raw}, 出廠年份: {input_year_val}, 行駛里程: {input_mileage_val} 公里 (車齡: {new_car_age} 年)")
    print(f"預測價格為: {predicted_price[0]:.2f} 萬元")
    print("==============================================")
except Exception as e:
    print(f"\n進行預測時發生錯誤: {e}")