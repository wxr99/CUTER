import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools

def fig0():
    plt.rcParams['font.family'] = ['serif']
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1,21)
    y = [ 19.,  19.,  24.,  19.,  27.,  25.,  47.,  36.,  71.,  37.,  42.,  51.,
         65.,  65., 205.,  69.,  71.,  84.,  94., 103.]
    y_sort = np.sort(y)[::-1]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(x))]
    plt.bar(x, y_sort, color=colors)
    plt.xticks(x, np.argsort(y)[::-1])
    plt.xlabel(r'Class')
    plt.ylabel(r'Number of samples')
    # ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.show()
    plt.savefig('./figures/num.pdf')

def fig1():
    x = [7.5,8.2,9.1,11.3,15.5,23.7,36.9,]
    y=[179.6752,179.58,176.5834,114.6570,64.1464,57.6549,43.2795]
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(x, y,linestyle='--', color='#8ECFC9', zorder=1, linewidth=1.5)
    plt.scatter(15.5, 64.1464, c='tan', marker='.', s=180, label='DINO v1')
    plt.scatter(23.7, 57.6549, c='mediumaquamarine', marker='.', s=180, label='DINO v2')
    plt.scatter(36.9, 43.2795, c='teal', marker='.', s=180, label='CUTLearn')
    plt.scatter(9.1, 176.5834, c='tomato', marker='.', s=180, label='MAE')
    plt.scatter(7.5, 179.6752, c='darkred', marker='.', s=180, label='iBOT')
    plt.scatter(11.3, 114.6570, c='olive', marker='.', s=180, label='MOCO v3')
    plt.scatter(8.2, 179.58, c='deepskyblue', marker='.', s=180, label='Supervised')

    plt.ylabel(r'average Fiedler Value', fontsize=12)
    plt.xlabel(r'zero-shot localization capacity ($AP_{50}$)', fontsize=12)
    ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.3)
    # plt.show()
    plt.savefig('./figures/fiedler_per.pdf')

def fig2():
    x = [7.5,8.2,9.1,11.3,15.5,23.7,36.9,]
    y=[82.3,60.4,67.6,62.8,66.4,81.1,72.4]
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(x, y,linestyle='--', color='#8ECFC9', zorder=1, linewidth=1.5)
    plt.scatter(15.5, 66.4, c='tan', marker='.', s=180, label='DINO v1')
    plt.scatter(23.7, 81.1, c='mediumaquamarine', marker='.', s=180, label='DINO v2')
    plt.scatter(36.9, 72.4, c='teal', marker='.', s=180, label='CUTLearn')
    plt.scatter(9.1, 67.6, c='tomato', marker='.', s=180, label='MAE')
    plt.scatter(7.5, 82.3, c='darkred', marker='.', s=180, label='iBOT')
    plt.scatter(11.3, 62.8, c='olive', marker='.', s=180, label='MOCO v3')
    plt.scatter(8.2, 60.4, c='deepskyblue', marker='.', s=180, label='Supervised')

    plt.ylabel(r'segmentation performance (mIoU)', fontsize=12)
    plt.xlabel(r'zero-shot localization capacity ($AP_{50}$)', fontsize=12)
    ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, framealpha=0.3)
    # plt.show()
    plt.savefig('./figures/ft_per.pdf')

def fig3():
    plt.rcParams['font.family'] = ['serif']
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1,21)
    y1 = [ 19.,  19.,  44.,  19.,  27.,  35.,  57.,  46.,  71.,  47.,  52.,  51.,
         75.,  75., 115.,  69.,  71.,  84.,  94., 103.]
    y3 = [53, 50, 50, 49, 49, 50, 48, 50, 51, 48, 52, 50, 49, 50, 52, 51, 50, 50, 58, 50,]
    y2 = [41, 36, 41, 44, 50, 46, 43, 32, 47, 43, 29, 43, 49, 36, 203, 35, 29, 49, 46, 48, ]
    width = 0.25
    plt.bar(x-width, y1, width, label='PRS', color='#BEB8DC')
    plt.bar(x, y2, width, label='OCDM', color='#FA7F6F')
    plt.bar(x+width, y3, width, label='CUTER', color='#82B0D2')
    plt.xticks(x, x, rotation=45, ha='right')
    plt.xlabel(r'Class')
    plt.ylabel(r'Number of samples')
    # ax.grid(True)
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    # plt.show()
    plt.savefig('./figures/balance.pdf')

def fig4():
    plt.rcParams['font.family'] = ['serif']
    fig, ax = plt.subplots(figsize=(5, 4))
    alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    plt.ylim(45, 85)
    voc = [79.45, 81.76, 81.97, 82.41, 82.02, 80.85]
    coco =[59.23, 59.81, 60.64, 60.08, 59.15, 59.42]
    nus = [50.51, 51.10, 51.43, 50.78, 50.62, 50.69]
    plt.plot(alpha, voc, linestyle='-.', marker='o', markersize=4, linewidth=2, color='#82B0D2', label='VOC')
    plt.plot(alpha, coco, linestyle='-.', marker='^', markersize=4, linewidth=2, color='#FFBE7A', label='MSCOCO')
    plt.plot(alpha, nus, linestyle='--', marker='*', markersize=4, linewidth=2, color='#8ECFC9', label='NUSWIDE')
    ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(r'$\alpha$', fontsize=14)
    plt.ylabel(r'$mAP$', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('./figures/sens1.pdf')
    # plt.show()


def fig5():
    plt.rcParams['font.family'] = ['serif']
    fig, ax = plt.subplots(figsize=(5, 4))
    tau1 = [0.50, 0.55, 0.60, 0.65, 0.70]
    plt.ylim(45, 85)
    voc = [82.34, 82.35, 81.74, 80.33, 80.55]
    coco =[59.87, 60.30, 59.85, 59.80, 59.01]
    nus = [50.92, 51.53, 51.03, 51.49, 50.08]
    plt.plot(tau1, voc, linestyle='-.', marker='o', markersize=4, linewidth=2, color='#82B0D2', label='VOC')
    plt.plot(tau1, coco, linestyle='-.', marker='^', markersize=4, linewidth=2, color='#FFBE7A', label='MSCOCO')
    plt.plot(tau1, nus, linestyle='--', marker='*', markersize=4, linewidth=2, color='#8ECFC9', label='NUSWIDE')
    ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(r'$\tau_1$', fontsize=14)
    plt.ylabel(r'$mAP$', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('./figures/sens2.pdf')
    # plt.show()

def fig6():
    plt.rcParams['font.family'] = ['serif']
    fig, ax = plt.subplots(figsize=(5, 4))
    tau2 = [0.7, 0.75, 0.8, 0.85, 0.9]
    plt.ylim(45, 85)
    voc = [81.98, 82.15, 82.03, 82.12, 81.86]
    coco =[59.45, 60.32, 60.08, 60.21, 59.95]
    nus = [50.43, 51.11, 51.36, 51.27, 50.96]
    plt.plot(tau2, voc, linestyle='-.', marker='o', markersize=4, linewidth=2, color='#82B0D2', label='VOC')
    plt.plot(tau2, coco, linestyle='-.', marker='^', markersize=4, linewidth=2, color='#FFBE7A', label='MSCOCO')
    plt.plot(tau2, nus, linestyle='--', marker='*', markersize=4, linewidth=2, color='#8ECFC9', label='NUSWIDE')
    ax.grid(True, color='white')
    ax.set_facecolor('#f2f2f2')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(r'$\tau_2$', fontsize=14)
    plt.ylabel(r'$mAP$', fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('./figures/sens3.pdf')
    # plt.show()

if __name__ == '__main__':
    # fig1()
    # fig2()
    # fig3()
    fig4()
    fig5()
    fig6()