import os
import pandas as pd
from sklearn.cluster import KMeans
import cv2
import numpy as np

def get_image_colors(image_path, num_colors=3):
    # 讀取圖片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # 使用KMeans算法對像素進行分群
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)

    # 獲取每個群集的顏色
    colors = kmeans.cluster_centers_

    return colors.astype(int)

# 讀取資料表
data = pd.read_csv("Clothes/Clothing_dataset_full/images.csv")

# 設定圖片資料夾路徑
image_folder = "Clothes/Clothing_dataset_full/images_original"

# 創建一個空的列表來存儲主要顏色
colors_list = []

# 遍歷所有圖片並獲取主要顏色
for index, row in data.iterrows():
    image_name = row["image"]
    image_path = os.path.join(image_folder, image_name+ ".jpg")
    print(image_path)
    # 使用前面提供的程式碼來獲取圖片的主要顏色
    colors = get_image_colors(image_path)

    # 添加主要顏色到列表中
    colors_list.append(colors)

# 將主要顏色添加到資料表中
data["main_colors"] = colors_list

data.to_csv("Clothes/Clothing_dataset_full/images.csv", index=False)
