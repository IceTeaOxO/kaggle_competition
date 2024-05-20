# 載入必要的庫
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
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
train_data = pd.read_csv("train_df.csv")

# 分割訓練集和驗證集
train_texts, train_labels = train_data["text"].tolist(), train_data["score"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# 將label轉換為數字
label_map = {"1 顆星": 1, "2 顆星": 2, "3 顆星": 3, "4 顆星": 4, "5 顆星": 5}
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]

# 初始化tokenizer和模型
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# 設定max_length
max_length = 55

# 創建訓練資料集和驗證資料集
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

# 設置訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
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
test_data = pd.read_csv("test_df.csv")
test_texts = test_data["text"].tolist()

# 預測score
predictions = []
for text in test_texts:
    # 在使用tokenizer時設定padding和max_length參數
    encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    output = model(**encoded_text)
    pred_label = torch.argmax(output.logits).item()
    predictions.append(pred_label)

# 儲存結果
result_df = pd.DataFrame({"index": test_data["index"], "pred": predictions})
result_df.to_csv("result.csv", index=False)
