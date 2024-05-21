import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./saved_model"  # 指定模型的儲存路徑

# 初始化tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

# 設定max_length
max_length = 55


test_data = pd.read_csv("Banking Apps Reviews Classification/test_preprocess_v3.csv")
test_data['text'].fillna('good', inplace=True)

test_texts = test_data["text"].tolist()
# 載入模型
loaded_model = RobertaForSequenceClassification.from_pretrained(model_path)
loaded_model.to(device)  # 將載入的模型移至 GPU

# 進行推論
predictions = []
for text in test_texts:
    encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    encoded_text = {key: val.to(device) for key, val in encoded_text.items()}  # 將數據移至 GPU
    output = loaded_model(**encoded_text)
    logits = output.logits
    probabilities = F.softmax(logits, dim=1)  # 將 logits 轉換為機率分佈
    pred_label = torch.argmax(probabilities, dim=1).item()  # 獲取預測標籤
    predictions.append(pred_label)

# 將預測結果轉換為顆星評分
pred_stars = [str(round(pred + 1)) + ' 顆星' for pred in predictions]

# 儲存推論結果
result_df = pd.DataFrame({"index": test_data["index"], "pred": pred_stars})
result_df.to_csv("inference_result.csv", index=False)
