import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report


# 讀取資料集
df = pd.read_csv('Banking Apps Reviews Classification/train_preprocess.csv')

# 填充缺失值為空字符串
df['text'].fillna('', inplace=True)

# 將評分轉換為數字類別
df['score'] = df['score'].map({'1 顆星': 1, '2 顆星': 2, '3 顆星': 3, '4 顆星': 4, '5 顆星': 5})

# 分割特徵和目標變量
X = df['text']
y = df['score']

# 使用TF-IDF向量化文本
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 將資料集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 設定類別加權
class_weights = {1: 0.15, 2: 0.25, 3: 0.2, 4: 0.25, 5: 0.15}

# 創建並訓練隨機森林模型
rf_model = RandomForestClassifier(class_weight=class_weights, n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 預測
y_pred = rf_model.predict(X_test)

# 評估模型性能
accuracy = accuracy_score(y_test, y_pred)
print(f"模型準確率：{accuracy}")
print("分類報告：")
print(classification_report(y_test, y_pred))


# 儲存模型和向量化器
import joblib
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')