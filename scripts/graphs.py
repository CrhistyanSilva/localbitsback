import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('stats_original.csv')
t = range(len(df))

for name in ['main_logd', 'z_logp', 'dequant_logd', 'total_logd']:
    s = -df[name].to_numpy()
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel='batch_id', ylabel='', title=name)
    plt.show()
    fig.savefig(f"{name}.png")
