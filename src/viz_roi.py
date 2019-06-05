import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os import listdir

def plot(data_df, out_path, metric='rank_mean'):
    plt.figure(figsize=(20,10))
    order = sorted(data_df['region'].unique())
    sns.barplot(data=data_df, x='region', y=metric, hue='subject', order=order)
    plt.xticks(rotation=45)
    plt.title('decoding %s by subject and ROI' % metric)
    plt.savefig(out_path, bbox_inches='tight')

def main():
    csv = [f for f in listdir('../results') if '.csv' in f]
    subjects, regions = zip(*[f.split('.')[3:5] for f in csv])
    subjects, regions = set(subjects), set(regions)
    data_dicts = []
    for s in subjects:
        for r in regions:
            path = '../results/perf.384sentences.baseline.{}.{}.csv'.format(s, r)
            decoder_results = pd.read_csv(path)
            data = {
                'subject' : s,
                'region' : r,
                'mse' : decoder_results['mse'],
                'r2' : decoder_results['r2'],
                'rank_mean' : decoder_results['rank_mean']
            }
            data_dicts.append(data)
    data_df = pd.DataFrame(data_dicts)
    metric = 'mse'
    plot(data_df, '../results/baseline_%s.png' % metric, metric=metric)

if __name__ == '__main__':
    main()