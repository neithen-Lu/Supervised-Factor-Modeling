import numpy as np
import pandas as pd
import itertools
import re
import matplotlib.pyplot as plt
import seaborn as sns

# for T0,T in itertools.product([25,50,100,200],[800]):
#     r=4
#     best_s=5
#     best_combo = f'({r};{r};5)'
#     df = pd.read_csv(f'../result/soft/soft_mix_1211/csv/rank{r}T{T+T0}T0{T0}.csv',header=0)
#     df.drop(columns=df.columns[-1:], axis=1, inplace=True)
#     selected_lmda = df.idxmin(axis=1)
#     print(selected_lmda.value_counts())

    

# T0=20
# df_full = pd.read_csv(f'../result/soft/soft_season_ar_1210/csv/est_rank{r}T{T+T0}T020.csv',header=None)
# df_full.drop(columns=df_full.columns[-1:], axis=1, inplace=True)
# for T0,T in itertools.product([40,60,80],[1000]):
#     # est err
#     df = pd.read_csv(f'../result/soft/soft_season_ar_1210/csv/est_rank{r}T{T+T0}T0{T0}.csv',header=None)
#     df.drop(columns=df.columns[-1:], axis=1, inplace=True)
#     df_full = pd.concat([df_full, df], axis=1)
# df_full.columns = ['20','40','60','80']
# sns.boxplot(df_full)
# plt.title('Estimation error vs T0 in soft thresholding')
# plt.savefig(f'soft_est_box.png')

# T = 1000;T0=20
# df_full = pd.read_csv(f'../result/soft/soft_season_ar_1210/csv/pred_rank{r}T{T+T0}T09.csv',header=None)
# df_full.drop(columns=df_full.columns[-1:], axis=1, inplace=True)
# for T0,T in itertools.product([40,60,80],[1000]):
#     # est err
#     df = pd.read_csv(f'../result/soft/soft_season_ar_1210/csv/pred_rank{r}T{T+T0}T0{T0}.csv',header=None)
#     df.drop(columns=df.columns[-1:], axis=1, inplace=True)
#     df_full = pd.concat([df_full, df], axis=1)
# df_full.columns = ['20','40','60','80']
# plt.figure()
# sns.boxplot(df_full)
# plt.title('Prediction error vs T0 in soft thresholding')
# plt.savefig(f'soft_pred_box.png')

for bic_c in np.arange(1e-3,1e-2,1e-3):
    print(bic_c)
    for T0,T in itertools.product([25,50,100],[800]):
        r=4
        df = pd.read_csv(f'../result/aic/soft_arma_1219/csv/rank{r}T{T}T0{T0}.csv',header=0)
        df.drop(columns=df.columns[-1:], axis=1, inplace=True)
        df_loss = df.copy(deep=True)
        for col in df.columns:
            try:
                r1,r2,s = re.findall(r'\d+', col)
            except:
                continue
            penalty = ((int(r1)+int(r2)) * 20 + np.log(T0))*int(s)/(T-T0)
            loss = df[col]
            df_loss[col] = np.log(loss)+bic_c*penalty
        selected_combo = df_loss.idxmin(axis=1)
        print('aic',selected_combo.value_counts())

for T0,T in itertools.product([25,50,100],[800]):
    df = pd.read_csv(f'../result/aic/soft_arma_1219/csv/est_rank{r}T{T}T0{T0}.csv',header=0)
    df.drop(columns=df.columns[-1:], axis=1, inplace=True)
    df = df.mean(axis=0)
    print('est',df)



        # rank
        # correct_rank_count = 0
        # s_count = [0,0,0]
        # for combo in selected_combo:
        #     r1,r2,s = re.findall(r'\d+', combo)
        #     s = int(s)
        #     if int(r1)==r and int(r2) == r:
        #         correct_rank_count += 1
        #     if s < best_s:
        #         s_count[0] += 1
        #     elif s == best_s:
        #         s_count[1] += 1
        #     elif s > best_s:
        #         s_count[2] += 1
        # print(correct_rank_count/len(selected_combo),s_count)

# df_full = pd.read_csv(f'../result/bic/soft_season_ar_1210/csv/est_rank{r}T{T+T0}T020.csv',header=0)
# df_full = df_full.iloc[:,13]
# for T0,T in itertools.product([40,60,80],[1000]):
#     # est err
#     df = pd.read_csv(f'../result/bic/soft_season_ar_1210/csv/est_rank{r}T{T+T0}T0{T0}.csv',header=0)
#     df = df.iloc[:,13]
#     df_full = pd.concat([df_full, df], axis=1)
# df_full.columns = ['20','40','60','80']
# plt.figure()
# sns.boxplot(df_full)
# plt.title('Estimation error vs T0 in hard thresholding')
# plt.savefig(f'hard_est_box.png')

# T = 1000;T0=20
# df_full = pd.read_csv(f'../result/bic/soft_season_ar_1210/csv/pred_rank{r}T{T+T0}T020.csv',header=0)
# df_full = df_full.iloc[:,13]
# for T0,T in itertools.product([40,60,80],[1000]):
#     # est err
#     df = pd.read_csv(f'../result/bic/soft_season_ar_1210/csv/pred_rank{r}T{T+T0}T0{T0}.csv',header=0)
#     df = df.iloc[:,13]
#     df_full = pd.concat([df_full, df], axis=1)
# df_full.columns = ['20','40','60','80']
# plt.figure()
# sns.boxplot(df_full)
# plt.title('Prediction error vs T0 in hard thresholding')
# plt.savefig(f'hard_pred_box.png')
