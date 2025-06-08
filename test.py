import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # 假設您用 MobileNetV2
import numpy as np
import matplotlib.pyplot as plt # 用於顯示圖片 (可選)

# --- 1. 設定參數 (需要與您訓練模型時的設定一致) ---
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
# 您的車型類別名稱列表，順序必須與模型訓練時 tf.data.Dataset 推斷的順序完全一致
# 通常可以從訓練時 train_dataset.class_names 獲得並儲存下來
# 【【【請務必將此列表替換成您實際的類別名稱和順序！】】】
CLASS_NAMES = ['Model_3', 'Model_S', 'Model_X', 'Model_Y'] # 範例，請修改！

# --- 2. 載入儲存的模型 ---
model_path = 'tesla_model_mobilenetv2.keras'
try:
    loaded_model = keras.models.load_model(model_path)
    print(f"模型 '{model_path}' 載入成功！")

except Exception as e:
    print(f"載入模型時發生錯誤: {e}")
    exit()

# --- 3. 準備要預測的新圖片 ---

new_image_path = 'test5.jpeg'

try:
    # 載入圖片並調整大小
    img = image.load_img(new_image_path, target_size=IMAGE_SIZE)
    
    # 將圖片轉換為 NumPy 陣列
    img_array = image.img_to_array(img)
    
    # 擴展維度，使其符合模型的輸入格式 (batch_size, height, width, channels)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # 進行與訓練時相同的預處理 (例如 MobileNetV2 的預處理)
    # 如果您的模型在建立時已經將 preprocess_input 作為第一層，這裡就不需要再做
    # 但通常 preprocess_input 是在模型之外處理的
    preprocessed_img = preprocess_input(img_array_expanded) # 針對 MobileNetV2
    
    print(f"\n圖片 '{new_image_path}' 預處理完成，準備進行預測。")

except FileNotFoundError:
    print(f"錯誤：找不到圖片檔案 '{new_image_path}'。請確認路徑是否正確。")
    exit()
except Exception as e:
    print(f"圖片預處理時發生錯誤: {e}")
    exit()

# --- 4. 使用載入的模型進行預測 ---
try:
    predictions = loaded_model.predict(preprocessed_img)

    print(f"\n模型原始預測輸出 (各類別機率): {predictions}")

    # --- 5. 解讀預測結果 ---
    # 找到機率最高的類別的索引
    predicted_class_index = np.argmax(predictions[0])
    
    # 根據索引從 CLASS_NAMES 列表中獲取類別名稱
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    # 獲取該類別的信心度 (機率值)
    confidence = predictions[0][predicted_class_index] * 100
    
    print("\n==============================================")
    print(f"預測結果：這張圖片最有可能是 {predicted_class_name}")
    print(f"信心度：{confidence:.2f}%")
    print("==============================================")

    #(可選) 顯示圖片和預測結果
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"進行預測或解讀結果時發生錯誤: {e}")