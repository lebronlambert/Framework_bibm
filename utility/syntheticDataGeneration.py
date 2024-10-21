__author__ = 'Haohan Wang and Liu Xiang'

import scipy
from simpleFunctions import *
import time
import random
class TreeNode:
    def __init__(self, k):
        self.node = k
        self.leftChild = None
        self.rightChild = None
        self.parent = None

class Tree:
    def __init__(self, kList):
        if len(kList)>0:
            self.root = TreeNode(kList[0])
            left = []
            right = []
            for i in range(1, len(kList)):
                if np.random.random() > 0.5:
                    left.append(kList[i])
                else:
                    right.append(kList[i])
            self.root.leftChild = Tree(left).root
            if self.root.leftChild is not None:
                self.root.leftChild.parent = self.root
            self.root.rightChild = Tree(right).root
            if self.root.rightChild is not None:
                self.root.rightChild.parent = self.root
        else:
            self.root = None

def generateLeftChildIndex(idx, p):
    l = idx.tolist()
    c = 0
    m = len(l)
    for i in range(p):
        if i not in idx:
            l.append(i)
            c += 1
            if c >= m:
                break
    return np.array(sorted(set(l)))

def generateRightChildIndex(idx):
    l = idx.tolist()
    return np.array(l[:-len(l)/2])

def generateTreeBeta(p, k, dense):
    beta = np.zeros([p, k])
    mask = np.zeros([p, k])
    indices = [[] for i in range(k)]
    tree = Tree(xrange(k))
    idx = np.array(sorted((scipy.random.randint(0,p,int(p*dense)).astype(int)).tolist()))
    queue = []
    indices[tree.root.node] = idx
    queue.append((tree.root.leftChild, 1))
    queue.append((tree.root.rightChild, 0))
    while len(queue)>0:
        c, f = queue[0]
        queue = queue[1:]
        pind = indices[c.parent.node]
        if f == 1:
            indices[c.node] = generateLeftChildIndex(pind, p)
        else:
            indices[c.node] = generateRightChildIndex(pind)
        if c.leftChild is not None:
            queue.append((c.leftChild, 1))
        if c.rightChild is not None:
            queue.append((c.rightChild, 0))
    for i in range(len(indices)):
        mask[indices[i], i] = np.ones([indices[i].shape[0]])
    for i in range(p):
        beta[i,:] = np.random.normal(0, 1, size=k) +i + 5
    beta = beta*mask
    return beta


def group_beta_generation(y_number, all_number):

    beta_matrix = np.zeros((all_number, y_number))
    group_num = int(np.log(y_number))

    group_size = []
    traits_left = y_number
    for i in range(group_num - 1):
        group_size.append(int(np.random.normal(traits_left / (group_num - i), 1, 1)))
        traits_left -= group_size[-1]
    group_size.append(traits_left)

    # For each group
    snp = 0
    traits = 0
    for i in range(group_num):
        snp_size = int(np.random.normal(2 * y_number, 1, 1))
        center = 15 + np.random.uniform(-3, 3, 1)
        for j in range(snp, snp + snp_size):
            beta_matrix[j][traits: traits + group_size[i]] = center
        snp += snp_size
        traits += group_size[i]

    # Between groups 0 & 1
    snp_size = int(np.random.normal(y_number / 1.5, 1, 1))
    center = 5 + np.random.uniform(-5, 5, 1)
    for j in range(snp, snp + snp_size):
        beta_matrix[j][0: group_size[0] + group_size[1]] = center * np.ones([1, group_size[0] + group_size[1]])
    snp += snp_size

    # Swap traits
    for i in range(int(5 * y_number)):
        a = np.random.randint(0, y_number, 1)
        b = np.random.randint(0, y_number, 1)
        beta_matrix[0:all_number, a], beta_matrix[0:all_number, b] = beta_matrix[0:all_number, b], beta_matrix[
                                                                                                   0:all_number, a]

    # Swap SNPs
    for i in range(int(5 * all_number)):
        a = np.random.randint(0, all_number, 1)
        b = np.random.randint(0, all_number, 1)
        beta_matrix[a], beta_matrix[b] = beta_matrix[b], beta_matrix[a]

    return beta_matrix



def generateData(n, p, g, d, k, sigX, sigY,we, tree="normal", test=False,lmmn="Lmm",quiet=False):
    time_start = time.time()
    sig = sigX
    sigC = sigY
    dense = d

    we =we
    center1 = np.random.uniform(0, 1, [g, p])
    sample = n / g
    X = []
    Z = []

    plt = None
    for i in range(g):
        x = np.random.multivariate_normal(center1[i, :], sig * np.diag(np.ones([p, ])), size=sample)
        X.extend(x)
    X = np.array(X)
    for i in range(g):
        for j in range(sample):
            Z.append(center1[i,:])

    Z = np.array(Z)
    # X[X > 1] = 1
    # X[X <= 1] = 0

    if tree=="tree":
        beta = generateTreeBeta(p, k, dense)
        beta_tmp=beta.copy()
        beta_tmp[beta_tmp != 0.] = -1
        beta_tmp[beta_tmp != -1] = 1
        beta_tmp[beta_tmp == -1] = 0
    elif tree=="group":
        beta=group_beta_generation(int(p*dense),p)
        beta_tmp=beta.copy()
        beta_tmp[beta_tmp != 0.] = -1
        beta_tmp[beta_tmp != -1] = 1
        beta_tmp[beta_tmp == -1] = 0
    elif tree=="normal":
        beta=np.zeros((p,k))
        for j in range(p):
            for k in range(k):
                if random.random()<=d:
                    beta[j,k]=10*random.random()  #random number 10
    else:
        print "no such generate way"
        exit(1)

    if test:
        from matplotlib import pyplot as plt
        plt.imshow(beta)
        plt.show()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.title.set_text('Simulate mapping linkage matrix')
        im1 = ax1.imshow(beta.T)
        plt.colorbar(im1, orientation='horizontal')
        plt.savefig('beta.eps', format='eps',dpi=50, figsize=(4, 4))
        plt.show()

    ypheno = X.dot(beta)

    error = np.random.normal(0, 1, ypheno.shape)

    C1 = np.dot(Z, Z.T)
    C = np.dot(X, X.T)
    if test:
        plt.imshow(C1)
        plt.savefig('C1.eps',format='eps')
        plt.show()

    if lmmn=="Lmm2":
        C1 = np.dot(C1, C1)
    if lmmn=="Lmmn":
        n = np.random.randint(1, 4)
        for i in range(n):
            C1 = np.dot(C1, C1)

    Kva, Kve = np.linalg.eigh(C)
    if test:
        ind = np.array(xrange(Kva.shape[0]))
        plt.scatter(ind[:-1], mapping2ZeroOne(Kva[:-1]), color='y', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 2)[:-1]), color='b', marker='+')
        plt.scatter(ind[:-1], mapping2ZeroOne(np.power(Kva, 4)[:-1]), color='m', marker='+')
        plt.savefig('K.png')
        plt.show()

    yK=[]
    for i in range(ypheno.shape[1]):
        yK_ = np.random.multivariate_normal(ypheno[:, i], sigC * C1, size=1)
        yK.extend(yK_)
    yK = np.array(yK)
    yK = yK.T
    yK = we * error + yK


    time_end = time.time()
    time_diff = time_end - time_start
    if quiet==False:
        print '%.2fs to generate data' % (time_diff)
    return X, yK, Kva, Kve, beta


if __name__=='__main__':
    pass


