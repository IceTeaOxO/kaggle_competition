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

def remove_numbers(text):
    # 去除數字
    text = re.sub(r'\d+', '', text)
    return text

def remove_characters(text):
    # 去除/, \, #, @, &, *, $, %, ^, (, ), [, ], {, }, <, >, |, `, ~, :, ;, ", ', =, +, -, _, \n
    text = re.sub(r'[/\\#@&*$%\/\[\]{}<>|`:;\'"，_]', '', text)
    return text

def remove_special_characters(text):
    # 去除標點符號和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    return text

import demoji

def replace_emojis_icon(text):
    # 找出文本中的表情符號
    emojis = demoji.findall(text)
    # 去除表情符號，並將表情符號的文字加入到文本後
    for emoji, emoji_text in emojis.items():
        text = text.replace(emoji, emoji_text)


    



    return text


def preprocess_text(text):
    # 將空值用 "good" 填入
    df['text'].fillna('good', inplace=True)
    
    # 分詞
    text = chinese_word_segmentation(text)
    # 去除停用詞
    # text = remove_chinese_stopwords(text)
    text = remove_numbers(text)
    # text = remove_special_characters(text)
    text = lowercase_text(text)
    # text = remove_characters(text)
    text = remove_stopwords(text)
    text = replace_emojis_icon(text)
    
    # text = stemming(text)
    # text = lemmatization(text)
    return text

def version6(text):
    text = replace_emojis_icon(str(text))
    text = remove_numbers(text)
    text = remove_characters(text)
    text = lowercase_text(text)
    df['text'].fillna('N/A', inplace=True)

    return text

df = pd.read_csv('Banking Apps Reviews Classification/test_df.csv')
# 找到包含空值的行
# rows_with_null = df[df['text'].isnull()]

# # 獲取包含空值的行的索引
# null_indexes = rows_with_null.index

# print(null_indexes)
# print(df.loc[null_indexes, 'text'])
# 對text欄位進行中文文本預處理
df['text'] = df['text'].apply(version6)
# 去除重複欄位資料,測試資料不能清理
# df.drop_duplicates(subset=['text'], inplace=True)
# 斷詞才需要清理空格
# df['text'] = df['text'].apply(lambda x: x.replace(' ', ''))


# 儲存預處理後的資料為train_preprocess.csv文件
df.to_csv('Banking Apps Reviews Classification/test_preprocess_v6.csv', index=False)

