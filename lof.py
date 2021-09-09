import numpy as np
from sklearn.neighbors import NearestNeighbors
import plotly.express as plt

"""
There are 3 code blocks to be filled to have the LOF algorithm run.
You should not need to alter the function interfaces.
"""
class LocalOutlierFactor(object):
    def __init__(self, MinPts, data):
        self.MinPts = MinPts
        self.data = data
        self.NumPoint = data.shape[0]

        # compute k-distance and build k-distance neighborhood
        self.nbrs = NearestNeighbors(n_neighbors=self.MinPts)
        self.nbrs.fit(data)
        [self.distNbs, self.indexNbs] = self.nbrs.kneighbors(data)
        self.kdist = np.amax(self.distNbs, axis=1)

        # initialize some results
        self.reachdist = self.distNbs
        self.lrd_data = np.zeros(self.NumPoint)
        self.lof_data = np.zeros(self.NumPoint)

        x_list = []
        y_list = []
        for one in range(self.data.shape[0]):
           x_list.append(self.data[one][0])
           y_list.append(self.data[one][1])
            
        fig = plt.scatter(x=x_list, y=y_list)
        fig.show()

    def ReachDist(self):
        for p in range(self.indexNbs.shape[0]):
            count = 0
            for i_o in range(self.indexNbs.shape[1]):
                o = self.indexNbs[p, i_o]

                """
                Code block: 1
                # compute reachability distance of an object p w.r.t. object o, stored in self.reachdist
                """

                # xo = self.data[o][0]
                # yo = self.data[o][1]
                # xp = self.data[p][0]
                # yp = self.data[p][1]

                # d = m.sqrt((xo - xp)**2 + (yo - yp)**2)

                
                d = self.reachdist[p][i_o]

                kd = self.reachdist[o][self.MinPts - 1]

                self.reachdist[p][i_o] = max(d, kd)


    def Lrd(self):

        """
        Code block: 2
        # compute local reachability density for each data point, stored in self.lrd_data
        """   

        for p in range(self.reachdist.shape[0]):
            sum = 0
            for o in range(self.reachdist.shape[1]):
                sum = sum + self.reachdist[p][o]

            self.lrd_data[p] = (self.MinPts/sum)


    def LofScore(self):

        """
        Code block: 3
        # compute the LOF score for each data point, stored in self.lof_data
        """    

        for p in range(self.lof_data.shape[0]):
            sum = 0
            for o in range(self.lof_data.shape[0]):
                sum = sum + (self.lrd_data[o]/self.lrd_data[p])
            
            self.lof_data[p] = sum/self.MinPts 

        return self.lof_data


    def LofAlgorithm(self):
        self.ReachDist()
        self.Lrd()
        self.LofScore()


def main():
    # Read data
    with open('data_1.npy', 'rb') as f:
        X = np.load(f).T

    # Set MinPts
    num_inst = X.shape[0]
    MinPtsRatio = 0.1
    MinPts = int(MinPtsRatio * num_inst)

    # Run LOF
    lof = LocalOutlierFactor(MinPts, X)
    lof.LofAlgorithm()

    # Show Results
    lof_scores = lof.lof_data
    IDs = lof_scores.argsort()[::-1][:3]
    for oid in IDs:
        print(X[oid])


if __name__ == "__main__":
    main()
