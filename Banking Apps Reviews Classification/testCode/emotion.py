from textblob import TextBlob

# 假設text是您的文本
text = "這是一個很好的產品，我很喜歡！"

# 使用TextBlob進行情感分析
blob = TextBlob(text)
sentiment_score = blob.sentiment.polarity  # 提取情感分數
print(sentiment_score)
# 將情感分數加入特徵
# 這裡僅是一個示例，您可以根據情感分析結果提取更多特徵
