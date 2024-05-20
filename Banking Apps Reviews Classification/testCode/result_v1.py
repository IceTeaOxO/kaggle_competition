import pandas as pd
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report
import re
import jieba
import pandas as pd
import nltk

def chinese_word_segmentation(text):
    # 使用結巴分詞進行中文分詞
    seg_list = jieba.cut(text, cut_all=False)
    segmented_text = ' '.join(seg_list)
    return segmented_text

def remove_chinese_stopwords(text):
    # 假設stopwords為停用詞列表
    stopwords = ['的', '了', '是', '在', '有', '和', '就', '這', '，', '您', '我']
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    text = ' '.join(filtered_words)
    return text


def remove_special_characters(text):
    # 去除標點符號和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    return text

def lowercase_text(text):
    # 將文本轉為小寫
    text = text.lower()
    return text

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def remove_stopwords(text):
    # 去除停用詞
    stop_words = set(stopwords.words('english'))  # 英文停用詞表
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

def stemming(text):
    # 詞幹提取
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    text = ' '.join(stemmed_words)
    return text

def lemmatization(text):
    # 詞形還原
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    return text

def preprocess_text(text):
    # 分詞
    text = chinese_word_segmentation(text)
    # 去除停用詞
    text = remove_chinese_stopwords(text)
    text = remove_special_characters(text)
    text = lowercase_text(text)
    text = remove_stopwords(text)
    text = stemming(text)
    text = lemmatization(text)
    return text
# 加載保存的隨機森林模型和TF-IDF向量化器
loaded_rf_model = joblib.load('random_forest_model.pkl')
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 讀取測試資料集
df_test = pd.read_csv('Banking Apps Reviews Classification/test_df.csv')

# 填充缺失值為空字符串
df_test['text'].fillna('', inplace=True)

# 對text欄位進行中文文本預處理
df_test['text'] = df_test['text'].apply(preprocess_text)

# 填充缺失值為空字符串
df_test['text'].fillna('', inplace=True)

# 使用向量化器對新的文本資料進行向量化
X_test = df_test['text']
X_test_tfidf = loaded_tfidf_vectorizer.transform(X_test)

# 使用加載的模型進行預測
predictions = loaded_rf_model.predict(X_test_tfidf)

# 生成結果DataFrame
result_df = pd.DataFrame({'index': df_test['index'], 'pred': predictions})
result_df['pred'] = result_df['pred'].astype(str) + ' 顆星'


# 將結果保存為result.csv文件
result_df.to_csv('result_v6.csv', index=False)
