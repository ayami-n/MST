import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from random import gauss
from scipy.spatial import distance

class MST:

    def __init__(self):
        self.data = pd.DataFrame(np.random.rand(50, 2))
        self.t_data = self.transform(self.data)
        # self.data = pd.DataFrame(np.array([[0, 1, 10], [0, 2, 6], [0, 3, 5], [1, 3, 15], [2, 3, 4]]))
        # self.data.rename(columns={0: "u", 1: "v", 2: "Weight"}, inplace=True)
        self.parent = []
        self.rank = []

    def transform(self, data):
        data = data.reset_index()
        data.rename(columns={"index": "ID", 0: "x", 1: "y"}, inplace=True)
        weight = []
        for itr in range(len(data)):  # for itr in  range(0 - 10)
            a = data.loc[itr, 'x'], data.loc[itr, 'y']
            for i in range(itr+1, len(data)):   # for i in range(1 - 10)
                b = data.loc[i, 'x'], data.loc[i, 'y']
                weight.append([data.loc[itr, 'ID'], data.loc[i, 'ID'], distance.euclidean(a, b)])
        df = pd.DataFrame(weight)  # make data frame
        df.rename(columns={0: "u", 1: "v", 2: "w"}, inplace=True)
        return df

    def viz(self, result):
        df = pd.DataFrame(result)
        for i, j in zip(df[0], df[1]):
            u = int(i)
            v = int(j)
            a = self.data.loc[u].values   # need original values
            b = self.data.loc[v].values
            plt.plot((a[0], b[0]),(a[1], b[1]), color='k')  # x = (a[0], b[0]), y = (a[1], b[1])
            plt.scatter((a[0], b[0]), (a[1], b[1]), s=50, color='r', marker='*')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.)
        plt.xlabel(r'$X$', size=14)
        plt.ylabel(r'$Y$', size=14)
        plt.tight_layout()
        plt.show()


    # disjoint set = make set + find set + merge
    def disjoint(self):
        i, j, edge = 0, 0, 0
        result = []
        # sorted by weight
        self.t_data = self.t_data.sort_values(by=['w'])
        # step1 makeSet -> point at itself
        self.makeSet()

        while edge < len(self.data) - 1: # must be num of vertices
            u, v, w = self.t_data.iloc[i]
            i = i + 1
            x = self.findSet(u)
            y = self.findSet(v)

            if x != y:
                edge = edge + 1
                result.append([u, v, w])
                self.union(x, y)

        min = 0
        for u, v, w in result:
            min += w
            print("%d -- %d == %f" % (u, v, w))
        print("Minimum Spanning Tree", min)

        self.viz(result)

    # version 3

    # step1 point at itself
    def makeSet(self):
        for node in range(len(self.data)):
            self.parent.append(node)
            self.rank.append(0)

    # step2 findSet
    def findSet(self, i):
        i = int(i)   # "i" is float type so, need float convert integer
        root = i
        while (root != self.parent[root]):
            root = self.parent[root]
        j = self.parent[i]
        while (j != root):
            self.parent[i] = root
            i = j
            j = self.parent[i]
        return root

    # step3 marge
    def merge(self, i, j):
        if (self.rank[i] < self.rank[j]):
            self.parent[i] = j
        elif (self.rank[i] > self.rank[j]):
            self.parent[j] = i
        else:
            self.parent[i] = j
            self.rank[j] = self.rank[j] + 1

    def union(self, i, j):
        self.merge(self.findSet(i), self.findSet(j))

if __name__ == '__main__':
    graph = MST()
    graph.disjoint()
