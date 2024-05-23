import pandas as pd
import numpy as np


# 讀取資料表
data = pd.read_csv("Banking Apps Reviews Classification/train_df.csv")

# 篩選出score為2 顆星的資料
# data = data[data["score"] == "1 顆星"]

# 加總text欄位的字數再除以所有text欄位的總數
data["text_length"] = data["text"].apply(lambda x: len(str(x)))

# 計算text欄位的平均字數
average_text_length = np.mean(data["text_length"])

print(average_text_length)
# 32.2

