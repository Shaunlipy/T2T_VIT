import pandas as pd 

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def process_df_in(df, colname):
    stats = {}
    for label in df['label'].unique():
        temp = df.loc[df['label']==label, colname]
        stats[label] = (temp.mean(), temp.std())
        print(f'label:{label},mean:{temp.mean():.5f},std:{temp.std():.5f}')


if __name__ == '__main__':
    ind_csv = '../inference_cifar10_cifar10_test.csv'
    ood_csv = '../inference_cifar10_imagenet_resize_ood.csv'
    ind_df = read_csv(ind_csv)
    process_df_in(ind_df, 'r_a')