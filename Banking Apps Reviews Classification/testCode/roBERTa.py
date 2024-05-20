# 載入必要的庫
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# 將文本和標籤轉換為模型可接受的格式
def preprocess_text(text, label):
    label_map = {"1 顆星": 1, "2 顆星": 2, "3 顆星": 3, "4 顆星": 4, "5 顆星": 5}
    label_id = label_map[label]
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(label_id)
    return inputs

# 讀取訓練資料
train_data = pd.read_csv("train_df.csv")

# 分割訓練集和驗證集
train_texts, train_labels = train_data["text"].tolist(), train_data["score"].tolist()
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

# 初始化tokenizer和模型
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

# 創建訓練資料集
train_dataset = []
for text, label in zip(train_texts, train_labels):
    inputs = preprocess_text(text, label)
    train_dataset.append(inputs)

# 創建驗證資料集
val_dataset = []
for text, label in zip(val_texts, val_labels):
    inputs = preprocess_text(text, label)
    val_dataset.append(inputs)

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
test_data = pd.read_csv("test_df.csv")
test_texts = test_data["text"].tolist()

# 預測score
predictions = []
for text in test_texts:
    # 在使用tokenizer時設定max_length參數
    # 在使用tokenizer時設定padding和max_length參數
    encoded_text = tokenizer(text, padding="max_length", max_length=55, truncation=True, return_tensors="pt")

    output = model(**encoded_text)
    pred_label = torch.argmax(output.logits).item()
    predictions.append(pred_label)

# 儲存結果
result_df = pd.DataFrame({"index": test_data["index"], "pred": predictions})
result_df.to_csv("result.csv", index=False)
