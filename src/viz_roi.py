import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os import listdir

ENCODINGS = sorted(['LM_pos', 'LM_scrambled', 'MNLI', 'QQP', 'SQuAD', 'SST'])
METRICS = ['mse', 'r2', 'rank_mean']

def plot(data_df, out_path, metric='rank_mean'):
    plt.figure(figsize=(20,10))
    order = sorted(data_df['region'].unique())
    hue_order = ENCODINGS
    sns.barplot(data=data_df, x='region', y=metric, hue='encoding', 
                order=order, hue_order=hue_order)
    plt.xticks(rotation=45)
    plt.title('decoding {} by ROI'.format(metric))
    plt.savefig(out_path, bbox_inches='tight')

def save_full_data_df():
    encoding_data = []
    data_dict = {
        'encoding' : [],
        'subject' : [],
        'region' : [],
        'mse' : [],
        'r2' : [],
        'rank_mean' : []
    }
    for encoding in ENCODINGS:
        print(encoding)
        csv = [f for f in listdir('../results/%s' % encoding) if '.csv' in f]
        subjects, regions = zip(*[f.split('.')[3:5] for f in csv])
        subjects, regions = set(subjects), set(regions)
        # data_dicts = []
        for s in subjects:
            for r in regions:
                path = '../results/{}/perf.384sentences.{}.{}.{}.csv'.format(encoding, encoding, s, r)
                decoder_results = pd.read_csv(path)
                data_dict['encoding'].append(encoding)
                data_dict['subject'].append(s)
                data_dict['region'].append(r)
                for metric in METRICS:
                    data_dict[metric].append(decoder_results[metric][0])   
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv('../results/full_data.csv', index=False)

def main():
    data_df = pd.read_csv('../results/full_data.csv')
    for metric in METRICS:
        print('plotting %s' % metric)
        plot(data_df, '../results/full_{}.png'.format(metric), metric=metric)

if __name__ == '__main__':
    main()