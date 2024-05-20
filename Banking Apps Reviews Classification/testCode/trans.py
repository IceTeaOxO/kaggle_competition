import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 讀取訓練資料
train_data = pd.read_csv('/content/train_preprocess_v3.csv')

# 將評分轉換為數字
train_data['score'] = train_data['score'].apply(lambda x: int(x.split()[0]) - 1)

# 初始化 tokenizer 和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)

# 定義 Dataset 類別
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        return {key: encoding[key].flatten() for key in encoding}, torch.tensor(label)

# 建立訓練集 Dataset 和 DataLoader
train_texts = train_data['text'].tolist()
train_labels = train_data['score'].tolist()
train_dataset = CustomDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 定義訓練參數
optimizer = AdamW(model.parameters(), lr=5e-5)

# 開始訓練
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    input_ids = batch[0]['input_ids']
    attention_mask = batch[0]['attention_mask']
    labels = batch[1]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# 儲存模型
model.save_pretrained('./model')
