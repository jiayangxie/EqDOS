from statsmodels.tsa.stattools import acf,pacf
import numpy as np
import os
import matplotlib.pyplot as plt

filename = 'surface/dos_all/0.npy'
doscar = np.load(os.path.join(filename))
x = doscar[:,0,:]
print(x.shape)
doscar_n = np.sum(doscar[:, 1:, :], 1)

plt.figure(figsize=(8, 5))
for i in range(x.shape[0]):
    plt.plot(x[i], doscar_n[i]/np.max(doscar_n[i]))
    plt.show()
    # lag_acf = acf(doscar_n[i], nlags=1500)
    # lag_acf = acf(np.diff(doscar_n[i]), nlags=1500)
    lag_acf = acf(np.diff(np.diff(doscar_n[i])), nlags=1500)
    plt.plot(lag_acf, marker='+')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(doscar_n[i])), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(doscar_n[i])), linestyle='--', color='gray')
    # 添加置信区间的灰色透明背景
    plt.fill_between(
        x=np.arange(len(lag_acf)),
        y1 = -1.96 / np.sqrt(len(doscar_n[i])),
        y2 = 1.96 / np.sqrt(len(doscar_n[i])),
        color='gray',
        alpha=0.05)
    plt.title('Autocorrelation Function')
    plt.xlabel('number of lags')
    plt.ylabel('correlation')
    plt.tight_layout()
plt.show()

# plt.figure(figsize=(8, 5))
# for i in range(x.shape[0]):
#     # plt.plot(x[i], doscar_n[i])
#     # plt.show()
#     # lag_pacf = pacf(doscar_n[i], nlags=100, method='ols')
#     # lag_pacf = pacf(np.diff(doscar_n[i]), nlags=100, method='ols')
#     lag_pacf = pacf(np.diff(np.diff(doscar_n[i])), nlags=100, method='ols')
#     plt.plot(lag_pacf, marker='+')
#     plt.axhline(y=0, linestyle='--', color='gray')
#     plt.axhline(y=-1.96 / np.sqrt(len(doscar_n[i])), linestyle='--', color='gray')
#     plt.axhline(y=1.96 / np.sqrt(len(doscar_n[i])), linestyle='--', color='gray')
#     plt.title('Autocorrelation Function')
#     plt.xlabel('number of lags')
#     plt.ylabel('correlation')
#     plt.tight_layout()
# plt.show()
