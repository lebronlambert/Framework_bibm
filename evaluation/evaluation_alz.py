__author__ = 'Haohan Wang'

import numpy as np

from utility.alzDataFunctions import *

modelNum = 9

def evaluate():
    snpNames = [line.strip().split('\t')[1] for line in open('../data/real/alz/marker.txt')]

    weights = np.load('../result/real/tree/beta_alz.npy')
    label = getLabels_alz()
    ind_l = np.where(label!=0)[0].tolist()

    i = 2
    weight = weights[i][:,-1]
    n = len(np.where(weight!=0)[0])
    ind = np.argsort(np.abs(weight))[::-1][:99]

    snps = []
    for i in ind.tolist():
        snps.append(snpNames[i])
        if i in ind_l:
            print i

    f = open('../result/alz20.csv', 'w')

    f.writelines('Rank,SNP,Rank,SNP,Rank,SNP\n')
    m = len(snps)/3
    for i in range(m):
        f.writelines(str(i+1)+','+snps[i]+','+str(i+m+1)+','+snps[i+m]+','+str(i+m+m+1)+','+snps[i+m+m]+'\n')
    f.close()



if __name__ == '__main__':
    evaluate()