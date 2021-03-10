import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve


def read_csv(file_path):
    dat = pd.read_csv(file_path)
    dat = dat.dropna()
    return dat

def hist_dt(dat_in, dat_out):
    plt.hist(dat_in.r_std, bins=50, density=True)
    plt.hist(dat_out.r_std, bins=50, density=True)
    plt.show()

def cal_auroc(dat_in, dat_out, col_name):
    y_score = dat_in[col_name].tolist() + dat_out[col_name].tolist()
    # print(y_score)
    y_true = [1.0]*len(dat_in[col_name]) + [0.0]*len(dat_out[col_name])
    # print(len(y_true))
    score = roc_auc_score(y_true, y_score)
    print(f'{col_name} auroc score: {score}')


if __name__ == '__main__':
    csv_in = '../inference_cifar10_cifar10_test.csv'
    csv_out = '../inference_cifar10_imagenet_resize_ood.csv'
    dat_in = read_csv(csv_in)
    dat_out = read_csv(csv_out)
    # hist_dt(dat_in, dat_out)
    # print(f'in-dist: {dat_in.r_m.mean():.5f},{dat_in.r_m.std():.5f};{dat_in.r_std.mean():.5f},{dat_in.r_std.std():.5f}')
    # print(f'OOD: {dat_out.r_m.mean():.5f},{dat_out.r_m.std():.5f};{dat_out.r_std.mean():.5f},{dat_out.r_std.std():.5f}')
    # file,label,pred,prob,r,r_5,r_a,r_a_5
    cal_auroc(dat_in, dat_out, 'prob')
    cal_auroc(dat_in, dat_out, 'r')
    cal_auroc(dat_in, dat_out, 'r_5')
    cal_auroc(dat_in, dat_out, 'r_a')
    cal_auroc(dat_in, dat_out, 'r_a_5')