import csv
import random

# 自行設定機率和對應的值
values = ['5 顆星', '4 顆星', '3 顆星', '2 顆星', '1 顆星']

with open('./Banking Apps Reviews Classification/test_df.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)

    # 創建新的CSV文件並寫入標題
    with open('result_v4.csv', mode='w', newline='') as output_file:
        fieldnames = ['index', 'pred']
        csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        # 遍歷原始CSV文件的每一行
        for row in csv_reader:
            text_length = len(row['text'])
            if text_length == 1:
                probabilities = [0.86, 0.06, 0.01, 0, 0.06]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 2:
                probabilities = [0.83, 0.10, 0.02, 0.01, 0.04]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 3:
                probabilities = [0.72, 0.10, 0.04, 0.02, 0.14]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 4:
                probabilities = [0.74, 0.08, 0.02, 0.02, 0.15]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 5:
                probabilities = [0.71, 0.08, 0.03, 0.02, 0.16]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 6:
                probabilities = [0.60, 0.08, 0.04, 0.02, 0.25]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 7:
                probabilities = [0.57, 0.07, 0.06, 0.03, 0.29]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 8:
                probabilities = [0.52, 0.06, 0.05, 0.04, 0.34]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 9:
                probabilities = [0.51, 0.06, 0.06, 0.05, 0.32]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 10:
                probabilities = [0.45, 0.06, 0.06, 0.04, 0.40]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 11:
                probabilities = [0.40, 0.08, 0.06, 0.06, 0.43]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 12:
                probabilities = [0.31, 0.06, 0.05, 0.08, 0.50]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 13:
                probabilities = [0.27, 0.06, 0.07, 0.10, 0.51]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 14:
                probabilities = [0.35, 0.06, 0.05, 0.04, 0.50]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 15:
                probabilities = [0.25, 0.05, 0.09, 0.10, 0.52]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 16:
                probabilities = [0.25, 0.08, 0.08, 0.07, 0.53]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 17:
                probabilities = [0.20, 0.08, 0.06, 0.04, 0.62]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 18:
                probabilities = [0.20, 0.08, 0.09, 0.10, 0.54]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 19:
                probabilities = [0.22, 0.05, 0.08, 0.10, 0.57]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 20:
                probabilities = [0.20, 0.06, 0.09, 0.09, 0.57]
                pred = random.choices(values, weights=probabilities)[0]
            elif text_length == 21:
                probabilities = [0.18, 0.06, 0.11, 0.10, 0.55]
                pred = random.choices(values, weights=probabilities)[0]
            else:
                # probabilities = [0.12, 0.06, 0.10, 0.10, 0.63]
                probabilities = [0, 0, 0, 0, 0.73]
                pred = random.choices(values, weights=probabilities)[0]

            # 寫入新的CSV文件
            csv_writer.writerow({'index': row['index'], 'pred': pred})

print("處理完成，結果已儲存在 result_v4.csv 文件中。")
