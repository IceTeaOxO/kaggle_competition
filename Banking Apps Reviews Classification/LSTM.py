import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

# 定義提早停止的回調函數
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 讀取預處理後的訓練資料
train_data = pd.read_csv('train_preprocess_v3.csv')
train_data['text'].fillna('good', inplace=True)

# 使用 Tokenizer 將文本轉換為序列
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(train_data['text'])
X_train = tokenizer.texts_to_sequences(train_data['text'])

# 將序列填充為相同的長度
max_len = 100
X_train = pad_sequences(X_train, maxlen=max_len)

# 建立 LSTM 模型
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(5, activation='softmax'))  # 輸出層改為 5 個單元，使用 softmax 激活函數

# 編譯模型
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 將評分轉換為 one-hot 編碼
y_train = pd.get_dummies(train_data['score'])

# 分割訓練集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 訓練模型
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 預測
# 請根據需要對測試資料進行特徵提取、序列填充，然後使用訓練好的模型進行預測


# 儲存 LSTM 模型
model.save('lstm_model.h5')



from keras.models import load_model

# 載入已儲存的 LSTM 模型
model = load_model('lstm_model.h5')

# 繼續訓練模型
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# 儲存更新後的模型
model.save('updated_lstm_model.h5')
