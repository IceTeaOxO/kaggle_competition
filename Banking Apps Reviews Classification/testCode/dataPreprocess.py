import pandas as pd


# 根據 app_name 的值進行分類
def G_app_name():
    df = pd.read_csv('Banking Apps Reviews Classification/train_df.csv')

    grouped = df.groupby('app_name')

    # 將分類後的資料寫入不同的 CSV 檔案
    for name, group in grouped:
        group.to_csv(f'./Banking Apps Reviews Classification/app_name/{name}_data.csv', index=False)

def G_text():
    df = pd.read_csv('Banking Apps Reviews Classification/train_df.csv')

    # 新增一列 'text_length'，存儲 'text' 列的字元長度
    # 新增一列 'text_length'，存儲 'text' 列的字元長度
    df['text_length'] = df['text'].apply(len)

    # 分類資料
    # df_0_7 = df[df['text_length'] <= 7]
    # df_7_20 = df[(df['text_length'] > 7) & (df['text_length'] <= 20)]
    # df_20_plus = df[df['text_length'] > 20]
    # df_0_6 = df[df['text_length'] <= 6]
    # df_0_8 = df[df['text_length'] <= 8]
    # df_7_15 = df[(df['text_length'] > 7) & (df['text_length'] <= 15)]
    # df_15_20 = df[(df['text_length'] > 15) & (df['text_length'] <= 20)]
    df_0 = df[df['text_length'] == 0]
    df_1 = df[df['text_length'] == 1]
    df_2 = df[df['text_length'] == 2]
    df_3 = df[df['text_length'] == 3]
    df_4 = df[df['text_length'] == 4]
    df_5 = df[df['text_length'] == 5]
    df_6 = df[df['text_length'] == 6]
    df_7 = df[df['text_length'] == 7]
    df_8 = df[df['text_length'] == 8]
    df_9 = df[df['text_length'] == 9]
    df_10 = df[df['text_length'] == 10]
    df_11 = df[df['text_length'] == 11]
    df_12 = df[df['text_length'] == 12]
    df_13 = df[df['text_length'] == 13]
    df_14 = df[df['text_length'] == 14]
    df_15 = df[df['text_length'] == 15]
    df_16 = df[df['text_length'] == 16]
    df_17 = df[df['text_length'] == 17]
    df_18 = df[df['text_length'] == 18]
    df_19 = df[df['text_length'] == 19]
    df_20 = df[df['text_length'] == 20]
    df_21 = df[df['text_length'] == 21]
    df_22_plus = df[df['text_length'] >= 22]


    # 儲存成不同的 CSV 檔案
    # df_0_7.to_csv('./Banking Apps Reviews Classification/text/0-7.csv', index=False)
    # df_7_20.to_csv('./Banking Apps Reviews Classification/text/7-20.csv', index=False)
    # df_20_plus.to_csv('./Banking Apps Reviews Classification/text/20+.csv', index=False)
    # df_0_6.to_csv('./Banking Apps Reviews Classification/text/0-6.csv', index=False)
    # df_0_8.to_csv('./Banking Apps Reviews Classification/text/0-8.csv', index=False)
    # df_7_15.to_csv('./Banking Apps Reviews Classification/text/7-15.csv', index=False)
    # df_15_20.to_csv('./Banking Apps Reviews Classification/text/15-20.csv', index=False)
    df_0.to_csv('./Banking Apps Reviews Classification/text/0.csv', index=False)
    df_1.to_csv('./Banking Apps Reviews Classification/text/1.csv', index=False)
    df_2.to_csv('./Banking Apps Reviews Classification/text/2.csv', index=False)
    df_3.to_csv('./Banking Apps Reviews Classification/text/3.csv', index=False)
    df_4.to_csv('./Banking Apps Reviews Classification/text/4.csv', index=False)
    df_5.to_csv('./Banking Apps Reviews Classification/text/5.csv', index=False)
    df_6.to_csv('./Banking Apps Reviews Classification/text/6.csv', index=False)
    df_7.to_csv('./Banking Apps Reviews Classification/text/7.csv', index=False)
    df_8.to_csv('./Banking Apps Reviews Classification/text/8.csv', index=False)
    df_9.to_csv('./Banking Apps Reviews Classification/text/9.csv', index=False)
    df_10.to_csv('./Banking Apps Reviews Classification/text/10.csv', index=False)
    df_11.to_csv('./Banking Apps Reviews Classification/text/11.csv', index=False)
    df_12.to_csv('./Banking Apps Reviews Classification/text/12.csv', index=False)
    df_13.to_csv('./Banking Apps Reviews Classification/text/13.csv', index=False)
    df_14.to_csv('./Banking Apps Reviews Classification/text/14.csv', index=False)
    df_15.to_csv('./Banking Apps Reviews Classification/text/15.csv', index=False)
    df_16.to_csv('./Banking Apps Reviews Classification/text/16.csv', index=False)
    df_17.to_csv('./Banking Apps Reviews Classification/text/17.csv', index=False)
    df_18.to_csv('./Banking Apps Reviews Classification/text/18.csv', index=False)
    df_19.to_csv('./Banking Apps Reviews Classification/text/19.csv', index=False)
    df_20.to_csv('./Banking Apps Reviews Classification/text/20.csv', index=False)
    df_21.to_csv('./Banking Apps Reviews Classification/text/21.csv', index=False)
    df_22_plus.to_csv('./Banking Apps Reviews Classification/text/22+.csv', index=False)


def G_score():
    df = pd.read_csv('Banking Apps Reviews Classification/train_df.csv')

    score_grouped = df.groupby('score')
    for name, group in score_grouped:
        group.to_csv(f'./Banking Apps Reviews Classification/score/{name}_data.csv', index=False)

# G_app_name()
G_text()
# G_score()