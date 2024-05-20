import csv
import requests
import time

# 定義Monica API的URL
monica_api_url = 'https://api.monica.com/api'

# 讀取CSV文件並處理
input_file = './Banking Apps Reviews Classification/test_df.csv'
output_file = 'result_v5.csv'

with open(input_file, mode='r', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    csv_reader = csv.DictReader(infile)
    fieldnames = ['index', 'pred']
    csv_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    for row in csv_reader:
        index = row['index']
        text = row['text']
        
        # 傳送請求給Monica API
        response = requests.post(monica_api_url, json={'index': index, 'text': text})
        
        # 檢查請求是否成功
        if response.status_code == 200:
            pred = response.json().get('pred')
        else:
            pred = 'Error'  # 如果請求失敗，您可以選擇如何處理這種情況

        # 寫入新的CSV文件
        csv_writer.writerow({'index': index, 'pred': pred})
        time.sleep(1)

print("處理完成，結果已儲存在 result_v1.csv 文件中。")
