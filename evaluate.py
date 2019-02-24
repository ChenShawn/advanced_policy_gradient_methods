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



class AgentEvaluator(object):
    records = []

    def evaluate(self, env, agent, num_episode=10, gamma=0.9, maxlen=200, qsize=2, render=True):
        ans = []
        for it in range(num_episode):
            acc_r = 0.0
            coeff = 1.0
            s0 = env.reset()
            que = [s0[None, :]]
            for it in range(qsize - 1):
                st, r, done, info = env.step(env.action_space.sample())
                que.append(st[None, :])
            for jt in range(maxlen):
                if render:
                    env.render()
                s = np.concatenate(que, axis=-1)
                a = agent.choose_action(s)
                st, r, done, _ = env.step(a)
                r = (r + 8.0) / 8.0
                que.pop(0)
                que.append(st[None, :])
                acc_r += coeff * r
                coeff *= gamma
                if done:
                    break
            ans.append(acc_r)
        print('Total reward in global step {}: {}'.format(agent.counter, ans[-1]))
        self.records.append(ans)


    def record_video(self, env, agent, qsize=2, maxlen=200):
        """record_video
        :param env: should be a gym.Env wrapped by Monitor
        :param record_dir: where to save the video
        """
        s = env.reset()
        que = [s[None, :]]
        for _ in range(qsize - 1):
            s, r, done, info = env.step(env.action_space.sample())
            que.append(s[None, :])
        for _ in range(maxlen):
            env.render()
            s = np.concatenate(que, axis=-1)
            a = agent.choose_action(s)
            s, r, done, _ = env.step(a)
            que.append(s[None, :])
            que.pop(0)
            if done:
                break


    def to_csv(self, csv_dir):
        df = pd.DataFrame()
        array = np.array(self.records, dtype=np.float32)
        df['step'] = np.arange(len(self.records), dtype=np.int32)
        df['reward_mean'] = array.mean(axis=-1)
        df['reward_std'] = array.std(axis=-1)
        df['reward_smooth'] = df['reward_mean'].ewm(span=20).mean()
        df['upper_bound'] = df['reward_mean'] + df['reward_std']
        df['lower_bound'] = df['reward_mean'] - df['reward_std']
        print(' [*] csv file successfully saved in ' + csv_dir)
        df.to_csv(csv_dir)


    def plot_from_csv(self, csv_dir, savefig=None):
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



if __name__ == '__main__':
    plot_from_csv(os.path.join('./logs/records/MountainCarContinuous-v0/', 'TNPG.csv'),
                  savefig=os.path.join('./logs/records/MountainCarContinuous-v0/TNPG.svg'))