import numpy as np
import pandas as pd
import itertools
import re
import matplotlib.pyplot as plt
import seaborn as sns

for aic_c in np.arange(1e-2,9e-2,1e-2):
    print(aic_c)
    # for r,T in itertools.product([4],[200,300,400,500,600,700]):
    #     T0=9
    #     best_s=5
    #     best_combo = f'({r};{r};5)'
    #     df = pd.read_csv(f'../result/aic/season_ar_full/csv/rank{r}T{T}.csv',header=0)
    for r,T in itertools.product([4],[500,1000,1500,2000]):
        if T == 500:
            best_s = 10; T0 = 33
        elif T == 1000:
            best_s = 11; T0 = 47
        elif T == 1500:
            best_s = 12; T0 = 58
        elif T == 2000:
            best_s = 12; T0 = 67
        # if T == 500:
        #     best_s = 8; T0 = 33
        # elif T == 1000:
        #     best_s = 9; T0 = 47
        # elif T == 1500:
        #     best_s = 10; T0 = 58
        # elif T == 2000:
        #     best_s = 10; T0 = 67
        try:
            df = pd.read_csv(f'../result/aic/sparserank_arma_1207/csv/rank{r}T{T}T0{T0}.csv',header=0)
        except:
            continue
        df.drop(columns=df.columns[-1:], axis=1, inplace=True)
        # df = df.iloc[:, 15:20]
        # df = df.iloc[:, 19:25]
        df_loss = df.copy(deep=True)
        for col in df.columns:
            try:
                r1,r2,s = re.findall(r'\d+', col)
            except:
                continue
            penalty = np.log(T-T0)*((int(r1)+int(r2)) * 10 + (np.log(T0)+int(r1)*int(r2))*int(s))/(T-T0)
            loss = df[col]
            df_loss[col] = np.log(loss)+aic_c*penalty
        selected_combo = df_loss.idxmin(axis=1)
        # print(selected_combo.value_counts())

        # rank
        correct_rank_count = 0
        s_count = [0,0,0]
        for combo in selected_combo:
            r1,r2,s = re.findall(r'\d+', combo)
            s = int(s)
            if int(r1)==r and int(r2) == r:
                correct_rank_count += 1
            if s < best_s:
                s_count[0] += 1
            elif s == best_s:
                s_count[1] += 1
            elif s > best_s:
                s_count[2] += 1
        print(correct_rank_count/len(selected_combo),s_count)


# for r,T in itertools.product([3],[1000]):
#     T0 = int(1.5*np.sqrt(T))
#     df = pd.read_csv(f'../result/aic/arma_try1202/csv/pred_rank{r}T{T} copy.csv',header=None)
#     df.drop(columns=df.columns[-1:], axis=1,  inplace=True)
#     pred_array = np.array(df).reshape((100,36))
#     sns.boxplot(pred_array[:,16:20])
#     plt.savefig('pred_box.png')

# for aic_c in np.arange(1e-2,9e-2,1e-2):
#     print(aic_c)
#     for T0 in [2,4,6,8]:
#         df = pd.read_csv(f'../result/money/csv/T0{T0}.csv',header=0)
#         # print(df.shape)
#         T=np.arange(215,243)
#         df.drop(columns=df.columns[-1:], axis=1,  inplace=True)
#         df_loss = df.copy(deep=True)
#         for col in df.columns:
#             T0,r1,r2,s = re.findall(r'\d+', col)
#             penalty = np.log(T-int(T0))*((int(r1)+int(r2)) * 10 + (np.log(int(T0))+int(r1)*int(r2))*int(s))/(T-int(T0))
#             loss = df[col]
#             df_loss[col] = np.log(loss)+aic_c*penalty
#         selected_combo = df_loss.idxmin(axis=1)
#         correct_count = 0
#         print(selected_combo.mode())

##### soft

# for aic_c in np.arange(1e-2,9e-2,1e-2):
#     print(aic_c)
#     for T0,T in itertools.product([9,12,15,18],[600]):
#         T0=9
#         r=4
#         best_s=5
#         best_combo = f'({r};{r};5)'
#         df = pd.read_csv(f'../result/aic/soft_season_ar_1208/csv/rank{r}T{T+T0}T0{T0}.csv',header=0)
#     # for r,T in itertools.product([4],[500,1000,1500,2000]):
#     #     if T == 500:
#     #         best_s = 10
#     #     elif T == 1000:
#     #         best_s = 11
#     #     elif T == 1500 or T == 2000:
#     #         best_s = 12
#     #     # best_s = (8,9,10)
#     #     T0 = int(1.5*np.sqrt(T))
#     #     try:
#     #         df = pd.read_csv(f'../result/aic/arma_1207/csv/rank{r}T{T}.csv',header=0)
#     #     except:
#     #         continue
#         df.drop(columns=df.columns[-1:], axis=1, inplace=True)
#         # df = df.iloc[:, 15:20]
#         # df = df.iloc[:, 19:25]
#         df_loss = df.copy(deep=True)
#         for col in df.columns:
#             try:
#                 r1,r2,s = re.findall(r'\d+', col)
#             except:
#                 continue
#             penalty = np.log(T-T0)*((int(r1)+int(r2)) * 10 + (np.log(T0)+int(r1)*int(r2))*int(s))/(T-T0)
#             loss = df[col]
#             df_loss[col] = np.log(loss)+aic_c*penalty
#         selected_combo = df_loss.idxmin(axis=1)
#         # print(selected_combo.value_counts())

#         # rank
#         correct_rank_count = 0
#         s_count = [0,0,0]
#         for combo in selected_combo:
#             r1,r2,s = re.findall(r'\d+', combo)
#             s = int(s)
#             if int(r1)==r and int(r2) == r:
#                 correct_rank_count += 1
#             if s < best_s:
#                 s_count[0] += 1
#             elif s == best_s:
#                 s_count[1] += 1
#             elif s > best_s:
#                 s_count[2] += 1
#         print(correct_rank_count/len(selected_combo),s_count)