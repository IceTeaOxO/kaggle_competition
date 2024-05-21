import pandas as pd

# 讀取資料集
df = pd.read_csv('Banking Apps Reviews Classification/results_csv/predicted_scores_modified.csv')

# 在 pred 欄位後面加上 " 顆星"
df['pred'] = df['pred'].astype(str) + ' 顆星'

# 儲存修改後的資料為新的 CSV 文件
df.to_csv('Banking Apps Reviews Classification/results_csv/predicted_scores_final.csv', index=False)
