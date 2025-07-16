import matplotlib.pyplot as plt

#TABLE1
plt.figure(figsize=(30,20))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['15', '25', '35', '45', '55']

# wifi_values
baseline_wifi = [89.67, 38.84, 26.97, 21.83, 16.25]
icarl_wifi = [90.41, 74.09, 66.9, 63.68, 63.49]
ucir_wifi = [88.07, 77.16, 69.24, 61.73, 63.58]
bic_wifi = [86.11, 64.91, 52.86, 50.1, 44.35]
our_wifi = [89.7, 88.69, 82.02, 76.65, 76.75]
89.44, 88.07, 82.7, 76.41, 77.15

plt.plot(x, baseline_wifi, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
plt.plot(x, icarl_wifi, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
plt.plot(x, ucir_wifi, color='yellowgreen', marker='^', markersize=20, linewidth=8)
plt.plot(x, bic_wifi, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
plt.plot(x, our_wifi, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
plt.xlabel("Number of Classes", fontsize=80, labelpad=30)
plt.ylabel("Accuracy(%)", fontsize=80)
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,100)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout()  # Adjust layout to make room for labels
plt.savefig("table1.png")
plt.clf()
#
#
##TABLE2
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{26}$-C$_{35}$', '+C$_{46}$-C$_{55}$', '+C$_{36}$-C$_{45}$', '+C$_{16}$-C$_{25}$']
## rfid_values
#baseline_rfid = [72.96, 27.40, 20.83, 16.81, 14.58]
#icarl_rfid = [74.15, 43.84, 29.89, 26.98, 21.43]
#ucir_rfid = [73.70, 42.24 ,30.84 ,29.93 , 25.03]
#bic_rfid = [73.81, 37.96, 31.73, 25.07, 23.84]
#our_rfid = [74.30, 52.31, 39.83, 34.64, 32.14]
#
##wifi_values
#baseline_wifi = [95.07, 37.02, 28.16, 20.46, 17.77]
#icarl_wifi = [94.85, 72.71, 63.24, 54.69, 54.1]
#ucir_wifi = [94.89 ,80.16 ,71.22 ,57.26 ,59.91]
#bic_wifi = [93.37, 72.09, 60.87, 57.89, 55.23]
#our_wifi = [95.07, 83.62, 76.00, 63.67, 64.36]
#
##mmwave_values 
#baseline_mmwave = [91.11, 35.2, 27.49, 20.47, 17.32]
#icarl_mmwave = [91.70, 73.22, 64.6, 55.7, 55.82]
#ucir_mmwave = [91.70, 73.22 ,64.6 ,55.7 ,55.82]
#bic_mmwave = [91.30, 77.0 ,70.49, 64.89, 63.83]
#our_mmwave = [93.07, 84.11, 78.7, 68.96, 70.77]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, baseline_rfid, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_rfid, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_rfid, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_rfid, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_rfid, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['Baseline', 'iCaRL', 'UCIR', 'BiC', 'Ours'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, baseline_wifi, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_wifi, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_wifi, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_wifi, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_wifi, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, baseline_mmwave, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_mmwave, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_mmwave, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_mmwave, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_mmwave, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table2.svg")
#
#plt.clf()
#
##TABLE3
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{36}$-C$_{45}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$']
#
##rfid_data
#baseline_rfid = [74.04, 32.02, 21.43, 18.46, 12.07]
#icarl_rfid = [73.74, 41.47, 30.03, 28.37, 23.28]
#ucir_rfid = [73.70, 49.53, 34.76, 31.19, 24.64]
#bic_rfid = [73.81, 37.67, 32.60, 27.90, 23.36]
#our_rfid = [74.30, 58.29, 45.21, 40.23, 33.86]
#
##wifi_data
#baseline_wifi = [95.07, 37.38, 28.3, 21.85, 17.14]
#icarl_wifi = [94.85, 73.2, 65.78, 58.83, 57.13]
#ucir_wifi = [94.89, 80.44, 71.19, 62.44, 61.12]
#bic_wifi = [93.37, 73.71, 62.95, 60.12, 54.03]
#our_wifi = [95.07, 83.4, 78.38, 66.72, 63.62]
#
##mmwave_data
#baseline_mmwave = [91.11, 36.8, 27.43, 21.35, 16.16]
#icarl_mmwave = [91.7, 72.4, 64.76, 61.35, 57.96]
#ucir_mmwave = [91.41, 75.36, 69.87, 67.81, 63.27]
#bic_mmwave = [91.3, 75.02, 70.22, 69.27, 65.24]
#our_mmwave = [93.07, 84.64, 78.41, 71.70, 69.78]
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, baseline_rfid, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_rfid, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_rfid, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_rfid, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_rfid, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['Baseline', 'iCaRL', 'UCIR', 'BiC', 'Ours'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, baseline_wifi, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_wifi, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_wifi, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_wifi, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_wifi, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, baseline_mmwave, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_mmwave, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_mmwave, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_mmwave, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_mmwave, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table3.svg")
#
#plt.clf()
#
##TABLE4
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$']
#
##rfid_data 
#baseline_rfid = [74.04, 28.91, 22.65, 14.63, 14.07]
#icarl_rfid = [73.74, 36.93, 27.25, 27.27, 23.59]
#ucir_rfid = [73.70, 44.20, 31.63, 34.09, 27.23]
#bic_rfid = [73.81, 40.18, 36.19, 27.47, 20.36]
#our_rfid = [74.30, 52.07, 42.21, 37.17, 33.15]
#
##wifi_data
#baseline_wifi = [95.07, 39.53, 27.98, 20.74, 17.12]
#icarl_wifi = [94.85, 72.96, 64.68, 57.62, 57.81]
#ucir_wifi = [94.89, 82.18, 70.9, 58.23, 60.01]
#bic_wifi = [93.37, 78.11, 63.46, 46.91, 43.55]
#our_wifi = [95.07, 84.82, 77.54, 62.19, 62.17]
#
##mmwave_data
#baseline_mmwave = [91.11, 38.60, 27.35, 19.60, 16.86]
#icarl_mmwave = [91.70, 74.38, 71.02, 62.10, 58.91]
#ucir_mmwave = [91.41, 76.96, 75.17, 67.31, 63.66]
#bic_mmwave = [91.30, 79.22, 73.06, 67.38, 60.16]
#our_mmwave = [93.07, 84.00, 82.59, 72.98, 70.37]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, baseline_rfid, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_rfid, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_rfid, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_rfid, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_rfid, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['Baseline', 'iCaRL', 'UCIR', 'BiC', 'Ours'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, baseline_wifi, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_wifi, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_wifi, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_wifi, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_wifi, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, baseline_mmwave, color=(115/255, 186/255, 214/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, icarl_mmwave, color=(13/255, 76/255, 109/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ucir_mmwave, color='yellowgreen', marker='^', markersize=20, linewidth=8)
#plt.plot(x, bic_mmwave, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8)
#plt.plot(x, our_mmwave, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table4.svg")
#
#plt.clf()
#
##TABLE5
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$', '+C$_{46}$-C$_{55}$']
##rfid_data
#baseline_rfid = [74.04, 43.53, 28.33, 28.07, 20.13]
#mse_rfid = [74.30, 48.20, 33.75, 28.59, 22.87]
#wa_rfid = [74.30, 48.80, 35.41, 37.27, 30.06]
#msewa_rfid = [75.96, 58.56, 44.89, 38.23, 34.01]
#
##wifi_data
#baseline_wifi = [95.07, 72.00, 65.57, 52.22, 56.01]
#mse_wifi = [94.85, 73.38, 66.89, 55.98, 56.87]
#wa_wifi = [95.07, 76.18, 69.40, 56.95, 61.08]
#msewa_wifi = [95.07, 83.56, 75.48, 64.12, 65.10]
#
##mmwave_data
#baseline_mmwave = [93.07, 76.38, 66.21, 57.96, 56.14]
#mse_mmwave = [93.07, 84.07, 73.97, 62.74, 59.45]
#wa_mmwave = [93.07, 80.91, 73.24, 60.72, 58.25]
#msewa_mmwave = [93.07, 87.49, 81.16, 69.35, 69.19]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, baseline_rfid, color=(135/255, 187/255, 164/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, mse_rfid, color=(158/255, 49/255, 80/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, wa_rfid, color=(229/255, 123/255, 127/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, msewa_rfid, color=(84/255, 104/255, 111/255), marker='p', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['Baseline', 'MSE', 'WA', 'MSE+WA'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, baseline_wifi, color=(135/255, 187/255, 164/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, mse_wifi, color=(158/255, 49/255, 80/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, wa_wifi, color=(229/255, 123/255, 127/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, msewa_wifi, color=(84/255, 104/255, 111/255), marker='p', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, baseline_mmwave, color=(135/255, 187/255, 164/255), marker='d', markersize=20, linewidth=8)
#plt.plot(x, mse_mmwave, color=(158/255, 49/255, 80/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, wa_mmwave, color=(229/255, 123/255, 127/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, msewa_mmwave, color=(84/255, 104/255, 111/255), marker='p', markersize=20, linewidth=8)
#plt.title('mmWave Rader', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table5.svg")
#
#plt.clf()
#
##TABLE6
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$']
##rfid_data
#resnet_rfid = [68.07, 43.09, 34.97, 27.04, 27.16]
#cbam_rfid = [75.15, 53.64, 43.79, 35.11, 34.41]
#unet_rfid = [75.96, 58.56, 44.90, 38.23, 34.01]
#
##wifi_data 
#resnet_wifi = [92.74, 78.76, 72.67, 62.11, 62.14]
#cbam_wifi = [92.63, 76.93, 72.49, 61.57, 63.38]
#unet_wifi = [95.07, 83.56, 75.48, 64.12, 65.10]
#
##mmwave_data 
#resnet_mmwave = [93.07, 87.49, 81.16, 69.35, 69.19]
#cbam_mmwave = [92.63, 78.36, 72.86, 62.05, 62.92]
#unet_mmwave = [92.96, 85.60, 77.19, 67.15, 65.37]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, resnet_rfid, color=(243/255, 162/255, 97/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, cbam_rfid, color=(138/255, 176/255, 125/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, unet_rfid, color=(75/255, 101/255, 175/255), marker='p', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['ResNet', 'CBAM', 'UNet'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, resnet_wifi, color=(243/255, 162/255, 97/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, cbam_wifi, color=(138/255, 176/255, 125/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, unet_wifi, color=(75/255, 101/255, 175/255), marker='p', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, resnet_mmwave, color=(243/255, 162/255, 97/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, cbam_mmwave, color=(138/255, 176/255, 125/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, unet_mmwave, color=(75/255, 101/255, 175/255), marker='p', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table6.svg")
#
#plt.clf()
#
##TABLE7
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$']
##rfid_data = {
#kld_rfid =  [72.96, 52.07, 40.65, 35.06, 29.53]
#l1_rfid = [74.04, 52.98, 38.56, 33.49, 29.29]
#mse_rfid =  [75.96, 58.56, 44.89, 38.23, 34.01]
#
##wifi_data = {
#kld_wifi = [95.07, 80.64, 73.67, 60.84, 61.97]
#l1_wifi = [95.07, 81.24, 74.06, 61.28, 63.37]
#mse_wifi = [95.07, 83.56, 75.48, 64.12, 65.10]
#
##mmwave_data = {
#kld_mmwave = [91.11, 82.0, 74.32, 64.69, 63.29]
#l1_mmwave = [91.11, 84.24, 77.67, 66.89, 66.70]
#mse_mmwave = [91.11, 84.31, 77.84, 69.19, 68.54]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, kld_rfid, color=(60/255, 64/255, 91/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l1_rfid, color=(130/255, 178/255, 154/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, mse_rfid, color=(223/255, 122/255, 94/255), marker='s', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['KLD', 'L1', 'MSE'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, kld_wifi, color=(60/255, 64/255, 91/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l1_wifi, color=(130/255, 178/255, 154/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, mse_wifi, color=(223/255, 122/255, 94/255), marker='s', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, kld_mmwave, color=(60/255, 64/255, 91/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l1_mmwave, color=(130/255, 178/255, 154/255), marker='^', markersize=20, linewidth=8)
#plt.plot(x, msewa_mmwave, color=(223/255, 122/255, 94/255), marker='s', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table7.svg")
#
#plt.clf()
#
##TABLE8
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$']
##rfid_data
#noex_rfid = [74.04, 33.24, 23.06, 19.41, 14.72]
#ex_rfid = [75.96, 58.56, 44.89, 38.23, 34.01]
##wifi_data
#noex_wifi = [95.07, 41.53, 33.76, 25.98, 24.2]      
#ex_wifi = [95.07, 83.56, 75.48, 64.12, 65.1]
##mmwave_data
#noex_mmwave = [91.11, 62.38, 47.86, 41.2, 35.29]       
#ex_mmwave = [91.11, 84.31, 77.84, 69.19, 68.54]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, noex_rfid, color=(75/255, 116/255, 178/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ex_rfid, color=(219/255, 49/255, 36/255), marker='^', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['No exemplar', 'Exemplar'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, noex_wifi, color=(75/255, 116/255, 178/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ex_wifi, color=(219/255, 49/255, 36/255), marker='^', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, noex_mmwave, color=(75/255, 116/255, 178/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, ex_mmwave, color=(219/255, 49/255, 36/255), marker='^', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table8.svg")
#
#plt.clf()
#
##TABLE9
#plt.figure(figsize=(50, 15))
#plt.rcParams['font.family'] = 'Times New Roman'
#x = ['C$_{1}$-C$_{15}$', '+C$_{46}$-C$_{55}$', '+C$_{16}$-C$_{25}$', '+C$_{26}$-C$_{35}$', '+C$_{36}$-C$_{45}$']
##rfid_data
#l1_rfid = [74.04, 52.31, 38.37, 29.98, 26.17]
#l2_rfid = [75.96, 58.56, 44.89, 38.23, 34.01]
##wifi_data
#l1_wifi = [95.07, 82.31, 75.11, 62.73, 64.71]      
#l2_wifi = [95.07, 83.56, 75.48, 64.12, 65.1]
##mmwave_data
#l1_mmwave = [91.11, 84.22, 77.67, 68.11, 67.15]       
#l2_mmwave = [91.11, 84.31, 77.84, 69.19, 68.54]
#
## Subplot 1 - RFID
#plt.subplot(1, 3, 1)
#plt.plot(x, l1_rfid, color=(145/255, 213/255, 66/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l2_rfid, color=(237/255, 104/255, 37/255), marker='^', markersize=20, linewidth=8)
#plt.title('RFID', fontsize=60)
#plt.legend(['L1', 'L2'], fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 2 - Wi-Fi
#plt.subplot(1, 3, 2)
#plt.plot(x, l1_wifi, color=(145/255, 213/255, 66/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l2_wifi, color=(237/255, 104/255, 37/255), marker='^', markersize=20, linewidth=8)
#plt.title('Wi-Fi', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
## Subplot 3 - mmWave Radar
#plt.subplot(1, 3, 3)
#plt.plot(x, l1_mmwave, color=(145/255, 213/255, 66/255), marker='o', markersize=20, linewidth=8)
#plt.plot(x, l2_mmwave, color=(237/255, 104/255, 37/255), marker='^', markersize=20, linewidth=8)
#plt.title('mmWave Radar', fontsize=60)
#plt.xlabel("Index of Classes", fontsize=60, labelpad=30)
#plt.ylabel("Accuracy(%)", fontsize=60)
#plt.xticks(fontsize=50)
#plt.yticks(fontsize=50)
#plt.grid(color='lightgray', linewidth=3)
#plt.ylim(0,100)
#bwith = 2 #边框宽度设置为2
#ax = plt.gca()#获取边框
#ax.spines['bottom'].set_linewidth(bwith)
#ax.spines['left'].set_linewidth(bwith)
#ax.spines['top'].set_linewidth(bwith)
#ax.spines['right'].set_linewidth(bwith)
#
#plt.tight_layout()  # Adjust layout to make room for labels
#plt.savefig("table9.svg")

#plt.clf()