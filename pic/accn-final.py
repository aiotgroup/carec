import matplotlib.pyplot as plt

#TABLE1
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
baseline_wifi = [89.67*0.01*15, 38.84*0.01*25, 26.97*0.01*35, 21.83*0.01*45, 16.25*0.01*55]
icarl_wifi = [90.41*0.01*15, 74.09*0.01*25, 66.9*0.01*35, 63.68*0.01*45, 63.49*0.01*55]
bic_wifi = [86.11*0.01*15, 64.91*0.01*25, 52.56*0.01*35, 50.1*0.01*45, 44.35*0.01*55]
ucir_wifi = [88.07*0.01*15, 77.16*0.01*25, 69.24*0.01*35, 61.73*0.01*45, 63.58*0.01*55]
beef_wifi = [89.81*0.01*15, 73.11*0.01*25, 63.46*0.01*35, 57.72*0.01*45, 53.73*0.01*55]
ccs_wifi = [87.48*0.01*15, 75.02*0.01*25, 66.59*0.01*35, 59.16*0.01*45, 60.45*0.01*55]
our_wifi = [89.44*0.01*15, 81.36*0.01*25, 75.0*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, baseline_wifi, color='y', marker='d', markersize=20, linewidth=8, label='Baseline')
plt.plot(x, icarl_wifi, color = (128/255, 0/255, 128/255), marker='o', markersize=20, linewidth=8, label='iCaRL')
plt.plot(x, ucir_wifi, color='yellowgreen', marker='^', markersize=20, linewidth=8, label='UCIR')
plt.plot(x, bic_wifi, color=(255/255, 158/255, 2/255), marker='p', markersize=20, linewidth=8, label='BiC')
plt.plot(x, beef_wifi, color='violet', marker='p', markersize=20, linewidth=8, label='BEEF')
plt.plot(x, ccs_wifi, color=(115/255, 186/255, 214/255), marker='p', markersize=20, linewidth=8, label='CCS')
plt.plot(x, our_wifi, color=(219/255, 49/255, 36/255), marker='s', markersize=20, linewidth=8, label='Ours')
plt.title('ACCN of Different CIL Method', fontsize=140)
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.legend(bbox_to_anchor=(0.5,0.96), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("accn1.png")
plt.clf()


#TABLE2
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
kd_wifi = [89.44*0.01*15, 81.18*0.01*25, 71.41*0.01*35, 62.12*0.01*45, 63.74*0.01*55]
bkd_wifi = [89.44*0.01*15, 81.36*0.01*25, 75*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, bkd_wifi, color=(219/255, 49/255, 36/255), marker='d', markersize=20, linewidth=8, label='BKD')
plt.plot(x, kd_wifi, color=(75/255, 116/255, 178/255), marker='o', markersize=20, linewidth=8, label='KD')
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.suptitle('ACCN of Different Knowledge Distillation', fontsize=140, x=0.15,y=0.95, horizontalalignment='left', va='bottom')
fig.legend(bbox_to_anchor=(0.5,0.93), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("kdbkd.png")
plt.clf()

#TABLE2
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
noex_wifi = [89.44*0.01*15, 39.22*0.01*25, 26.35*0.01*35, 21.36*0.01*45, 15.96*0.01*55]
ex_wifi = [89.44*0.01*15, 81.36*0.01*25, 75*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, noex_wifi, color=(223/255, 122/255, 94/255), marker='d', markersize=20, linewidth=8, label='No Examplar')
plt.plot(x, ex_wifi, color=(130/255, 178/255, 154/255), marker='o', markersize=20, linewidth=8, label='Examplar')
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.suptitle('ACCN with or without Data Replay', fontsize=140, x=0.2,y=0.95, horizontalalignment='left', va='bottom')
fig.legend(bbox_to_anchor=(0.6,0.93), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("examplar.png")
plt.clf()


#TABLE2
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
none_wifi = [89.44*0.01*15, 75.02*0.01*25, 66.59*0.01*35, 59.16*0.01*45, 60.45*0.01*55]
onlyer_wifi = [89.44*0.01*15, 82.02*0.01*25, 74.63*0.01*35, 64.37*0.01*45, 66.15*0.01*55]
eraux_wifi = [89.44*0.01*15, 81.36*0.01*25, 75*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, none_wifi, color=(138/255, 176/255, 125/255), marker='d', markersize=20, linewidth=8, label='DR')
plt.plot(x, onlyer_wifi, color=(243/255, 162/255, 97/255), marker='o', markersize=20, linewidth=8, label='DR+ER')
plt.plot(x, eraux_wifi, color=(75/255, 101/255, 175/255), marker='o', markersize=20, linewidth=8, label='DR+ER+AUX')
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.suptitle('ACCN of Model Expansion Module', fontsize=140, x=0.2,y=0.95, horizontalalignment='left', va='bottom')
fig.legend(bbox_to_anchor=(0.6,0.93), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("expand.png")
plt.clf()


#TABLE2
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
noco_wifi = [89.44*0.01*15, 88.07*0.01*25, 78.40*0.01*35, 72*0.01*45, 70.09*0.01*55]
co_wifi = [89.44*0.01*15, 81.36*0.01*25, 75*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, noco_wifi, color=(145/255, 213/255, 66/255), marker='d', markersize=20, linewidth=8, label='No Compression')
plt.plot(x, co_wifi, color=(237/255, 104/255, 37/255), marker='o', markersize=20, linewidth=8, label='Compression')
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.suptitle('ACCN with or without Compression', fontsize=140, x=0.2,y=0.95, horizontalalignment='left', va='bottom')
fig.legend(bbox_to_anchor=(0.75,0.93), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("compress.png")
plt.clf()


#TABLE2
fig=plt.figure(figsize=(45, 35))
plt.rcParams['font.family'] = 'Times New Roman'
x = ['Session 0', 'Session 1', 'Session 2', 'Session 3', 'Session 4']

# wifi_values
ideal_wifi = [95.07*0.01*15, 95.07*0.01*25, 95.07*0.01*35, 95.07*0.01*45, 95.07*0.01*55]
resnet_wifi = [87.8*0.01*15, 72.8*0.01*25, 64.54*0.01*35, 53.14*0.01*45, 57.72*0.01*55]
unet_wifi = [89.44*0.01*15, 81.36*0.01*25, 75*0.01*35, 63.46*0.01*45, 67.84*0.01*55]

plt.plot(x, ideal_wifi, color='gray', marker='o', markersize=20, linewidth=8, label='Ideal')
plt.plot(x, resnet_wifi, color='chocolate', marker='d', markersize=20, linewidth=8, label='ResNet')
plt.plot(x, unet_wifi, color='yellowgreen', marker='o', markersize=20, linewidth=8, label='UNet')
plt.xlabel("Incremental Session", fontsize=140, labelpad=30)
plt.ylabel("ACCN", fontsize=140)
plt.xticks(fontsize=140,rotation=30)
plt.yticks(fontsize=120)
plt.grid(color='lightgray', linewidth=3)
plt.ylim(0,60)
bwith = 2 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

fig.suptitle('ACCN of Different Backbone', fontsize=140, x=0.25,y=0.95, horizontalalignment='left', va='bottom')
fig.legend(bbox_to_anchor=(0.53,0.93), ncol=2, fontsize= 100,columnspacing=0.4)
fig.tight_layout()  # Adjust layout to make room for labels
plt.savefig("backbone.png")
plt.clf()