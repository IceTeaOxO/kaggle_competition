# 載入必要的庫
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

# 自定義Dataset類別
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # 去掉batch維度
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs

# 讀取訓練資料
train_data = pd.read_csv("Banking Apps Reviews Classification/train_preprocess_v3.csv")
train_data['text'].fillna('good', inplace=True)
# 分割訓練集和驗證集
train_texts, train_labels = train_data["text"].tolist(), train_data["score"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# 將label轉換為數字
label_map = {"1 顆星": 0, "2 顆星": 1, "3 顆星": 2, "4 顆星": 3, "5 顆星": 4}
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]


# 初始化tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

# 設定max_length
max_length = 55

# 創建訓練資料集和驗證資料集
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

# 將模型和數據移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_dataset.to(device)
val_dataset.to(device)

# 設置訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# 定義訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 開始訓練
trainer.train()

# 進行推論
test_data = pd.read_csv("Banking Apps Reviews Classification/test_preprocess_v3.csv")
test_data['text'].fillna('good', inplace=True)

test_texts = test_data["text"].tolist()

import torch.nn.functional as F
predictions = []
for text in test_texts:
    encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    output = model(**encoded_text)
    logits = output.logits
    probabilities = F.softmax(logits, dim=1)  # 將 logits 轉換為機率分佈
    pred_label = torch.argmax(probabilities, dim=1).item()  # 獲取預測標籤
    predictions.append(pred_label)
# 預測score
# predictions = []
# for text in test_texts:
#     # 在使用tokenizer時設定padding和max_length參數
#     encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
#     output = model(**encoded_text)
#     pred_label = torch.argmax(output.logits).item()
#     predictions.append(pred_label)

# 儲存結果
result_df = pd.DataFrame({"index": test_data["index"], "pred": predictions})
result_df.to_csv("result.csv", index=False)
