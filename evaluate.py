import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import os


COLORS = ['orange', 'blue', 'red', 'green', 'black', 'cyan', 'yellow']

def plot_from_csv(csv_dir, savefig=None):
    df = pd.read_csv(csv_dir)

    mstyle.use('ggplot')
    plt.figure()
    plt.plot(np.arange(df['reward_mean'].__len__()), df['reward_smooth'], color='orange')
    plt.fill_between(np.arange(df['reward_smooth'].__len__()),
                     df['upper_bound'],
                     df['lower_bound'],
                     color='orange', alpha=0.2)
    plt.ylabel('Averaged reward')
    plt.xlabel('Steps of env interaction')
    plt.title('TNPG on {}'.format(csv_dir.split('.')[0]))
    if savefig is not None and type(savefig) is str:
        plt.savefig(savefig, format='svg')
    plt.show()


def plot_from_files(dir, savefig=None):
    """plot_from_files
    :param dir: type str representing the folder of csv files
    :param savefig: whether to save figure (default svg format)
    """
    fs = os.listdir(dir)
    csvs = [name for name in fs if '.csv' in name]
    files = [pd.read_csv(os.path.join(name)) for name in csvs]

    mstyle.use('ggplot')
    plt.figure()
    for color, df in zip(COLORS, files):
        plt.plot(np.arange(df['reward_mean'].__len__()), df['reward_smooth'], color=color)
        plt.fill_between(np.arange(df['reward_smooth'].__len__()),
                         df['upper_bound'],
                         df['lower_bound'],
                         color=color, alpha=0.2)
        plt.ylabel('Averaged reward')
        plt.xlabel('Steps of env interaction')

    plt.legend([name.split('.')[0] for name in csvs])
    plt.title('Performance on {}'.format(dir.split('/')[-1]))
    if savefig is not None and type(savefig) is str:
        plt.savefig(savefig, format='svg')
    plt.show()



if __name__ == '__main__':
    plot_from_csv(os.path.join('./logs/records/MountainCarContinuous-v0/', 'TNPG.csv'),
                  savefig=os.path.join('./logs/records/MountainCarContinuous-v0/TNPG.svg')