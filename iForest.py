import matplotlib.pyplot as plt
import numpy as np
import random as rand
from collections import Counter

"""
There are 3 code blocks to be filled to have the LOF algorithm run.
You should not need to make changes to other code segments.
You should not need to alter the function interfaces.
"""
class Node:
    def __init__(self, val=None, right=None, left=None):
        self.val = val  # store the IDs of data points, only for leaf nodes
        self.right = right
        self.left = left


class iTree:

    def __init__(self):
        self.tree = None
        self.n_features = None

    def get_split(self, data, inds):

        """
        Code block: 1
        # randomly choose (1) which feature f to split, (2) the feature split value v
        """

        f = rand.randrange(data[0].shape[0])        

        x_list = []
        y_list = []

        for each in data:
            x_list.append(each[0])
            y_list.append(each[1])

        max_x = max(x_list)
        min_x = min(x_list)
        max_y = max(y_list)
        min_y = min(y_list)

        if f == 0:
            rand_val = rand.uniform(min_x, max_x)
        else:
            rand_val = rand.uniform(min_y, max_y)

        v = rand_val

        return f, v

    def split(self, data, IDs):
        # split the current set of data points
        f, v = self.get_split(data, IDs)

        left_ind = []
        right_ind = []
        for i in IDs:

            if data[i, f] <= v:
                left_ind.append(i)
            else:
                right_ind.append(i)

        return left_ind, right_ind

    def buildTree(self, data, IDs, count = 0):
        
        """
        Code block: 2
        # build the non-leaf node, which recursively makes the tree deeper
        """
        # leaf node
        if len(IDs) <2 or count > 25:
            return Node(val=IDs)

        # non-leaf node
        else:

            left_ind, right_ind = self.split(data, IDs)

            left = self.buildTree(data, left_ind, count = count + 1)
            right = self.buildTree(data, right_ind, count = count + 1)

            return Node(left=left, right=right)
        

    def fit(self, data):
        self.n_features = data.shape[1]
        IDs = np.arange(data.shape[0])
        self.tree = self.buildTree(data, IDs)
        return

    def traverse(self):
        # traver the tree, compute the path length
        path_len = Counter()
        i = -1

        def helper(currentNode):
            nonlocal i
            i += 1
            if currentNode.val is not None:
                for ind in currentNode.val:
                    path_len[ind] = i
                return
            for child in [currentNode.left, currentNode.right]:
                helper(child)
                i -= 1
            return

        helper(self.tree)
        return path_len


class iForest:
    def __init__(self, n_tree, ratio_outlier):
        self.n_tree = n_tree
        self.ratio_outlier = ratio_outlier  # ratio of outliers in the dataset
        self.outlierScores = Counter()

    def fit_predict(self, data):
        n_sample = data.shape[0]


        sum_list = []

        for _ in range(self.n_tree):
            itree = iTree()
            itree.fit(data)                 # build each iTree
            path_len = itree.traverse()     # compute path length
            self.outlierScores = self.outlierScores + path_len

            """
            Code block: 3
            # compute the normalization factor c_phi
            """    

            sum = 0
            for each in path_len:
                sum = sum + path_len.get(each)

            sum_avg = sum/n_sample

            sum_list.append(sum_avg)

        sum = 0
        for each in sum_list:
            sum = sum + each

        c_phi = sum / len(sum_list)

        # normalization factor
        c_phi = sum/n_sample

        for key, val in self.outlierScores.items():
            self.outlierScores[key] = 2 ** -(val / self.n_tree / c_phi)
        q = np.quantile(list(self.outlierScores.values()), 1 - self.ratio_outlier)
        outliers = [key for key, val in self.outlierScores.items() if val > q]
        return outliers


if __name__ == '__main__':
    with open('data_2.npy', 'rb') as f:
        X = np.load(f)

    IF = iForest(100, 0.02)
    outierIDs = IF.fit_predict(X)
    outliers = X[outierIDs]
    for oid in outierIDs:
        print(X[oid])

    plt.scatter(X[:, 0], X[:, 1], color='b')
    plt.scatter(outliers[:, 0], outliers[:, 1], color='r')
    plt.show()
