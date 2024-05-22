import pandas as pd

# 讀取包含預測值的CSV文件
df = pd.read_csv("roBERTa_v5.csv")

# 定義映射關係（這裡假設0對應1顆星，1對應2顆星，以此類推）
mapping = {0: "1 顆星", 1: "2 顆星", 2: "3 顆星", 3: "4 顆星", 4: "5 顆星"}

# 將"pred"列的值映射為1到5顆星
df['pred'] = df['pred'].map(mapping)

# 將處理後的DataFrame保存為新的CSV文件
df.to_csv("roBERTa_v5_star.csv", index=False)
