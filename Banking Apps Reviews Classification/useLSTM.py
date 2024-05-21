import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 讀取測試資料
test_data = pd.read_csv('/content/test_preprocess_v3.csv')
test_data['text'].fillna('good', inplace=True)

# 使用 Tokenizer 將文本轉換為序列
X_test = tokenizer.texts_to_sequences(test_data['text'])
X_test = pad_sequences(X_test, maxlen=max_len)

# 加載訓練好的模型
model = load_model('lstm_model.h5')  # 請替換成您訓練好的模型的路徑

# 預測
predictions = model.predict(X_test)

# 將預測結果轉換為顆星評分
pred_stars = [str(round(pred.argmax() + 1)) + ' 顆星' for pred in predictions]

# 創建包含預測結果的 DataFrame
result_df = pd.DataFrame({'index': test_data['index'], 'pred': pred_stars})

# 儲存預測結果為 result.csv 文件
result_df.to_csv('result_v2.csv', index=False)
