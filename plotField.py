import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


def plotField(data, xlabel = None, ylabel = None, title = None, ax = None, ylim = None, xlim = None, **kwargs):
    ax = ax or plt.gca()
    ax.title(title, size = 36)
    ax.xlabel(xlabel, size = 30)
    ax.ylabel(ylabel, size = 30)
    ax.ylim(ylim)
    ax.xlim(xlim)
    ax.grid(color = 'b', alpha = 0.5, linestyle = 'dashed', linewidth = 0.5)
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)
    return ax.plot(data, **kwargs)

x = np.array((0, 1, 2, 3))
y = np.array((4, 5, 6, 7))

plt.figure(figsize = (12, 8))
plotField(data = x, ax = plt, label = 'Estimated', color = 'r', linewidth = '3', linestyle='dashed')
plotField(data = y, ax = plt, label = 'True', color = 'b', linewidth = '3', ylabel = 'Oi', xlabel = 'oi', title = 'Ai', ylim = [0, 1])
plt.legend(fontsize = 28, loc = "lower left")
plt.show()