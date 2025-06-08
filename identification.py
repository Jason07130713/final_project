import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt # 用於顯示圖片或訓練曲線 (可選)
import numpy as np



# --- 1. 設定參數 ---
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BATCH_SIZE = 32 # 如果記憶體不足，可以嘗試調小，例如 16
NUM_CLASSES = 4  # 您有 Model S, 3, X, Y 四種車型
EPOCHS = 20      # 初始訓練的輪數，可以根據情況調整
LEARNING_RATE = 0.001

DATASET_PATH = 'D:/dataset'
# --- 2. 載入與準備數據 ---
print("--- 載入數據 ---")
# 使用 image_dataset_from_directory 從資料夾載入圖片
# 它會自動根據資料夾名稱推斷標籤 (labels)
# 並將數據分割為訓練集和驗證集 (80% 訓練, 20% 驗證)
try:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,  # 20% 的數據作為驗證集
        subset="training",
        seed=123,  # 設定隨機種子以確保可重現性
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int' # 標籤是整數 (0, 1, 2, 3)
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
except Exception as e:
    print(f"錯誤：載入數據時發生問題。請檢查 DATASET_PATH ('{DATASET_PATH}') 是否正確，")
    print(f"並且該路徑下是否確實存在 Model_S, Model_3, Model_X, Model_Y 這四個子資料夾，且裡面有圖片。")
    print(f"詳細錯誤訊息: {e}")
    exit()

class_names = train_dataset.class_names
print(f"辨識的車型類別: {class_names}")
if len(class_names) != NUM_CLASSES:
    print(f"警告：在資料夾中找到的類別數量 ({len(class_names)}) 與設定的 NUM_CLASSES ({NUM_CLASSES}) 不符。請檢查！")
    # 您可以選擇在這裡 exit() 或讓程式繼續，但模型輸出層可能需要調整

print(f"訓練集批次數量: {len(train_dataset)}")
print(f"驗證集批次數量: {len(validation_dataset)}")

# 優化數據載入性能 (可選，但推薦)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# --- 3. 建立資料增強層 ---
print("\n--- 建立資料增強層 ---")
data_augmentation = Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),
        layers.RandomRotation(0.1), # 隨機旋轉 +/- 10%
        layers.RandomZoom(0.1),   # 隨機縮放 +/- 10%
        # layers.RandomContrast(0.1), # 隨機調整對比度 (可選)
    ],
    name="data_augmentation",
)

# --- 4. 建立模型 (使用 MobileNetV2 進行遷移學習) ---
print("\n--- 建立模型 ---")
# 載入預訓練的 MobileNetV2 模型 (不包含頂層的分類器)
#input_shape 設定為我們圖片的大小，weights='imagenet' 表示使用在 ImageNet 上預訓練的權重
base_model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                         include_top=False,  # 不載入原始模型的頂層分類器
                         weights='imagenet')

# 凍結預訓練模型的權重，這樣在初始訓練時它們不會被更新
base_model.trainable = False

# 建立我們自己的模型
inputs = keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
x = data_augmentation(inputs)           # 1. 應用資料增強
x = preprocess_input(x)                 # 2. MobileNetV2 需要的特定預處理
x = base_model(x, training=False)       # 3. 運行預訓練模型 (training=False 很重要，因為我們凍結了它的層)
x = layers.GlobalAveragePooling2D()(x)  # 4. 將特徵圖轉換為向量
x = layers.Dropout(0.2)(x)              # 5. 加入 Dropout 層防止過度擬合 (可選)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x) # 6. 我們的輸出層 (4個Tesla車型, softmax激活)

model = keras.Model(inputs, outputs)

# --- 5. 編譯模型 ---
print("\n--- 編譯模型 ---")
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy', # 因為 label_mode='int'
              metrics=['accuracy'])

model.summary() # 印出模型結構

# --- 6. 訓練模型 ---
print("\n--- 開始訓練模型 ---")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint('best_tesla_model.keras', save_best_only=True, monitor='val_loss')

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint] # 加入 Callbacks
)

print("模型訓練完成！")

#--- 7. (初步) 微調 (Fine-tuning) - 可選的進階步驟 ---
#在初始訓練幾輪後，可以解凍 MobileNetV2 的部分頂層，用更小的學習率進行微調
base_model.trainable = True
# 通常只解凍較高層的卷積層，例如最後幾十層
fine_tune_at = 100 # 假設解凍 MobileNetV2 從第 100 層開始之後的層
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10), # 用更小的學習率
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- 開始微調模型 ---")
history_fine = model.fit(
    train_dataset,
    epochs=EPOCHS + 10, # 再多訓練幾輪
    initial_epoch=history.epoch[-1] + 1, # 從上次結束的 epoch 開始
    validation_data=validation_dataset
    # callbacks=[model_checkpoint] # 可以繼續使用 ModelCheckpoint
)
print("模型微調完成！")


# --- 8. 模型評估與儲存 (後續步驟) ---
print("\n--- 模型評估與儲存 (後續步驟) ---")


model.save('tesla_model_mobilenetv2.keras')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

if 'history_fine' in locals(): # 如果進行了微調
    acc += history_fine.history['accuracy']
    val_acc += history_fine.history['val_accuracy']
    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


