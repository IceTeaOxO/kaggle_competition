import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

max_length = 55

# 載入模型和tokenizer
model_path = "./results/checkpoint-10500"  # 指定之前訓練好的模型的路徑
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 將模型移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 準備推論數據
test_data = pd.read_csv("Banking Apps Reviews Classification/test_preprocess_v3.csv")
test_data['text'].fillna('good', inplace=True)

test_texts = test_data["text"].tolist()

# 進行推論
import torch.nn.functional as F
predictions = []
for text in test_texts:
    encoded_text = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    encoded_text = {key: val.to(device) for key, val in encoded_text.items()}  # 將數據移至與模型相同的設備
    output = model(**encoded_text)
    logits = output.logits
    probabilities = F.softmax(logits, dim=1)  # 將 logits 轉換為機率分佈
    pred_label = torch.argmax(probabilities, dim=1).item()  # 獲取預測標籤
    predictions.append(pred_label)

result_df = pd.DataFrame({"index": test_data["index"], "pred": predictions})
result_df.to_csv("roBERTa_v5.csv", index=False)
