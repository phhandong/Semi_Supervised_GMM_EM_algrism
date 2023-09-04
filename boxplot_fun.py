import matplotlib.pyplot as plt
#画 M 列boxplot图
#n 为图表title名
#data 为[N * M]的numpy矩阵
#label 为[1 * M]的numpy矩阵，作为x轴的label
#fig_dir 为储存的地址
def draw_boxplot(n,data,label,fig_dir):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    plt.title(n)
    colors = ['#8ECFC9','#FFBE7A','#FA7F6F','#82B0D2','#BEB8DC']
    bp = ax.boxplot(data, labels=label, meanline=True, notch=True, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=0.7)
    for cap in bp['caps']:
        cap.set(color='#8B008B',
                linewidth=2)
    for flier in bp['fliers']:
        flier.set(marker='D',
                  color='#e7298a',
                  alpha=0.5)
    fig.savefig(fig_dir + '/' + n + ".png")
