import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback


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
train_data = pd.read_csv("Banking Apps Reviews Classification/train_preprocess_v7.csv")
train_data['text'].fillna('good', inplace=True)
# 分割訓練集和驗證集
train_texts, train_labels = train_data["text"].tolist(), train_data["score"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# 將label轉換為數字
label_map = {"1 顆星": 0, "2 顆星": 1, "3 顆星": 2, "4 顆星": 3, "5 顆星": 4}
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]

# 初始化tokenizer和模型（使用BERT）
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=5)
# model = BertForSequenceClassification.from_pretrained("./model/saved_model_BERT_chinese_v5", num_labels=5)


# 設定max_length
max_length = 50

# 將模型移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 創建訓練資料集和驗證資料集
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)

# 設置訓練參數，包括 Early Stopping
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="steps",  # 每隔多少步驟進行一次驗證
    eval_steps=500,  # 每隔500步驟進行一次驗證
    load_best_model_at_end=True,  # 在訓練結束時載入最佳模型
)

# 定義 Early Stopping Callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)  # 如果性能連續3次沒有改善，則停止訓練

# 定義訓練器，加入 Early Stopping Callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping]  # 加入 Early Stopping Callback
)

# 開始訓練
trainer.train()

# 進行推論
test_data = pd.read_csv("Banking Apps Reviews Classification/test_preprocess_v6.csv")
test_data['text'].fillna('N/A', inplace=True)

test_texts = test_data["text"].tolist()

import torch.nn.functional as F
predictions = []
for text in test_texts:
    encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    encoded_text = {key: val.to(device) for key, val in encoded_text.items()}  # 將數據移至 GPU
    output = model(**encoded_text)
    logits = output.logits
    probabilities = F.softmax(logits, dim=1)  # 將 logits 轉換為機率分佈
    pred_label = torch.argmax(probabilities, dim=1).item()  # 獲取預測標籤
    predictions.append(pred_label)

# 將預測結果轉換為顆星評分
pred_stars = [str(round(pred + 1)) + ' 顆星' for pred in predictions]

# 儲存推論結果
result_df = pd.DataFrame({"index": test_data["index"], "pred": pred_stars})
result_df.to_csv("inference_result_BERT_chinese_v7_v2.csv", index=False)

# 儲存訓練後的模型
model_path = "./model/saved_model_BERT_chinese_v7_v2"  # 指定模型的儲存路徑
trainer.save_model(model_path)
