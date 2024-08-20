

import os
# os.environ['PYTHONHASHSEED'] = '0'

import matplotlib
havedisplay = "DISPLAY" in os.environ
if havedisplay:  #if you have a display use a plotting backend
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import networkx as nx


def plot_histogram(data, xlabel, ylabel, filename, ifhist=True, ifrug=False, ifkde=False, ifxlog=False, ifylog=False, figsize=(15,10), color="cornflowerblue"):
    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.distplot(data, hist=ifhist, rug=ifrug, kde=ifkde, color=color)
    if ifxlog:
        plt.xscale("log")
    if ifylog:
        plt.yscale("log")  #plt.yscale("log",basey=10), where basex or basey are the bases of log

    plt.tick_params(labelsize=30)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font1 = {'family':'DejaVu Sans','weight':'normal','size':30,}
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    output_path = '/home/xzhang/workplace/SCTCCI/AUPRC'
    plt.savefig('{}/{}.png'.format(output_path, filename))
    plt.close()


def plot_evaluating_metrics(metrics_list, fig_xlabel, fig_ylabel, fig_legend, filename, figsize=(13,9), color='orange'):
    figure, ax = plt.subplots(figsize=figsize, dpi=100)

    # ax = plt.subplot(111, facecolor='linen')

    # 多项式拟合
    f1 = np.polyfit(range(1,len(metrics_list)+1), np.array(metrics_list), 6)
    p1 = np.poly1d(f1)
    yvals1 = p1(range(1,len(metrics_list)+1))
    # 指定函数拟合
    # def func(x,a,b,c):  #指数函数拟合
    #     return a*np.exp(b/x)+c
    # def func(x,a,b,c):  #非线性最小二乘法拟合
    #     return a*np.sqrt(x)*(b*np.square(x)+c)
    # popt1, pcov1 = curve_fit(func, range(1,len(metrics_list)+1), np.array(metrics_list))  #popt里面是拟合系数：a=popt[0]，b=popt[1]，c=popt[2]
    # yvals1 = func(range(1,len(metrics_list)+1), *popt1)

    ax.plot(range(1,len(metrics_list)+1), yvals1, color=color, linestyle='-', linewidth=6)

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font1 = {'family':'DejaVu Sans','weight':'normal','size':30,}
    plt.xlabel(fig_xlabel, font1)
    plt.ylabel(fig_ylabel, font1)

    font2 = {'family':'DejaVu Sans','size':25,}
    leg = plt.legend(fig_legend, bbox_to_anchor=(1.02, 0), loc='lower right', borderaxespad=0, prop=font2)
    leg.get_frame().set_linewidth(0.0)
    output_path = '/home/xzhang/workplace/VGAECCI-main/VGAE-CCI/AUPRC'
    plt.savefig('{}/{}.png'.format(output_path, filename))     #.tif本来是
    plt.close()


def plot_cluster_score(cluster_num, score, xlabel, ylabel, filename, line_mode="bx-"):
    figure, ax = plt.subplots(figsize=(15.36,7.67), dpi=100)

    plt.plot(cluster_num, score, line_mode)

    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font1 = {'family':'DejaVu Sans','weight':'normal','size':30,}
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)

    figure.subplots_adjust(right=0.75)

    plt.savefig('{}.png'.format(filename))    #tif
    plt.close()


def plot_spatial_cluster(cluster_label, coord, filename, figsize=(15.36,7.67)):
    colors = ['beige', 'royalblue', 'maroon', 'olive', 'tomato', 'mediumpurple', 'paleturquoise', 'brown', 
              'firebrick', 'mediumturquoise', 'lightsalmon', 'orchid', 'dimgray', 'dodgerblue', 'mistyrose', 
              'sienna', 'tan', 'teal', 'chartreuse']

    X = np.hstack((cluster_label, coord))

    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    for cluster in np.unique(X[:,1]):
        plt.scatter(X[X[:,1] == cluster, 2], X[X[:,1] == cluster, 3], color=colors[int(cluster)], s=38, alpha = 1, label='D'+str(cluster))

    figure.subplots_adjust(right=0.67)
    plt.tick_params(labelsize=38)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('DejaVu Sans') for label in labels]

    font1 = {'family':'DejaVu Sans','weight':'normal','size':38,}
    plt.xlabel('x coordinates', font1)
    plt.ylabel('y coordinates', font1)

    font2 = {'family':'DejaVu Sans','size':38,}
    plt.legend(bbox_to_anchor=(1.005, 0), loc=3, borderaxespad=0, prop=font2)

    plt.savefig(filename + '.png')   











