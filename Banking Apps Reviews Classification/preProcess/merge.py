import pandas as pd

# 讀取包含'pred'的CSV文件
df_pred = pd.read_csv('inference_result_BERT_chinese_v7_v3.csv')

# 讀取包含'text'的CSV文件
df_text = pd.read_csv('Banking Apps Reviews Classification/test_preprocess_v7.csv')

# 根據'index'將兩個DataFrame進行合併
merged_df = pd.merge(df_pred, df_text, on='index', how='left')

merged_df.to_csv('answer_v7_v3.csv', index=False)
# 顯示合併後的DataFrame
# print(merged_df)
