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

def remove_duplicates():
    # 去除重複資料
    df = pd.read_csv('Banking Apps Reviews Classification/train_preprocess_v2.csv')
    df = df.drop_duplicates(subset=['text'])
    df.to_csv('Banking Apps Reviews Classification/train_preprocess_v3.csv', index=False)


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

df = pd.read_csv('Banking Apps Reviews Classification/train_df.csv', encoding='utf-8', errors='replace')

# 對text欄位進行中文文本預處理
df['text'] = df['text'].apply(preprocess_text)

# 去除重複資料
remove_duplicates()

# 儲存預處理後的資料為train_preprocess.csv文件
# df.to_csv('Banking Apps Reviews Classification/train_preprocess.csv', index=False)

