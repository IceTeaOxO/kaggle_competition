import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback
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
train_data = pd.read_csv("Banking Apps Reviews Classification/train_preprocess_v5.csv")
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
max_length = 35

# 將模型移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 創建訓練資料集和驗證資料集
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length)



# 載入上次的checkpoint
model_path = "./results/checkpoint-11000"  # 上次訓練的checkpoint路徑
model = RobertaForSequenceClassification.from_pretrained(model_path)

# 設置訓練參數
training_args = TrainingArguments(
    output_dir="./results_continue_training",  # 新模型的輸出目錄
    num_train_epochs=5,  # 訓練輪數
    per_device_train_batch_size=8,  # 每個設備的訓練批次大小
    learning_rate=5e-5,  # 學習率
    warmup_steps=500,  # 預熱步數
    weight_decay=0.01,  # 權重衰減
    logging_dir="./logs",  # 日誌目錄
    evaluation_strategy="steps",  # 每隔多少步進行一次驗證
    eval_steps=500,  # 每隔多少步進行一次驗證
)

# 創建訓練器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 開始訓練
trainer.train()
