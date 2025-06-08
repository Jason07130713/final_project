import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df_original = pd.read_csv('tesla.csv')
df = df_original.copy()



df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)

df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.dropna(subset=['age'], inplace=True)

df['mile'] = pd.to_numeric(df['mile'], errors='coerce')
df.dropna(subset=['mile'], inplace=True)

df['model'].fillna('Unknown', inplace=True)
df = pd.get_dummies(df, columns=['model'], prefix='Model', drop_first=True)

print(f"預處理後用於模型的數據筆數: {len(df)}")


y = df['price']
X = df.drop('price', axis=1)
if 'year' in X.columns:
    X = X.drop('year', axis=1)

X_input_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"訓練集筆數: {len(X_train)}")
print(f"測試集筆數: {len(X_test)}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n隨機森林迴歸模型評估結果：")
print(f"  平均絕對誤差 (MAE): {mae_rf:.2f} 萬元")
print(f"  均方根誤差 (RMSE): {rmse_rf:.2f} 萬元")
print(f"  R 平方分數 (R-squared): {r2_rf:.2f}")

while True:
    try:
        input_model_name_raw = input(f"請輸入車型 (例如 Model_S, Model_3, Model_X, Model_Y)").strip()
        if not input_model_name_raw:
            print("程式結束。")
            break

        input_year_val = int(input("請輸入出廠年份 (例如 2022): "))
        input_mileage_val = float(input("請輸入行駛里程 (公里，例如 30000): "))
        
        new_car_age = 2025 - input_year_val
        
        new_car_input_df = pd.DataFrame(columns=X_input_columns)
        new_car_input_df.loc[0] = 0
        
        new_car_input_df.loc[0, 'age'] = new_car_age
        new_car_input_df.loc[0, 'mile'] = input_mileage_val
        

        model_ohe_column_to_set = f"Model_{input_model_name_raw}"
        if model_ohe_column_to_set in X_input_columns:
            new_car_input_df.loc[0, model_ohe_column_to_set] = 1
        
        predicted_price = rf_model.predict(new_car_input_df)
        

        print("\n---------------- 預測結果 ----------------")
        print(f"對於 車型: {input_model_name_raw}, 出廠年份: {input_year_val}, 行駛里程: {input_mileage_val} 公里")
        print(f"模型預測價格為: {predicted_price[0]:.2f} 萬元")
        print("------------------------------------------")

    except ValueError:
        print("\n輸入錯誤！年份和里程必須是數字，請重新輸入。")
        continue
    except Exception as e:
        print(f"\n發生預期外的錯誤: {e}")
        break
