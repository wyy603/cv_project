import numpy as np
import matplotlib.pyplot as plt
import pickle, re

def get():
    loss = []
    acc = []
    with open("log/model_3/qwq.txt", "r") as f:
        lines = f.read().split('\n')
        for i in range(63):
            loss.append(float(re.findall(r'\d+\.\d+', lines[2 * i])[0]))
            acc.append(float(re.findall(r'\d+\.\d+', lines[2 * i + 1])[0]))
    return loss, acc

def draw():
    #with open("../log/model/100_data.pkl", "rb") as f:
    #    loss, acc = pickle.load(f)

    epochs = np.arange(1, 64)
    loss, acc = get()

    # 创建图像
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 acc 线
    ax1.plot(epochs, acc, color='b', label='Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_yticks(np.arange(0, 1.1, 0.1))

    # 创建第二个 y 轴用于绘制 loss
    ax2 = ax1.twinx()

    # 绘制 loss 线
    ax2.plot(epochs, loss, color='r', label='Loss')
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params('y', colors='r')
    #ax2.set_yticks(np.arange(0, 1.1, 0.1))

    # 添加标题和网格
    plt.title('Training Metrics')
    plt.grid(True)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.savefig("doc/fig.png")

draw()