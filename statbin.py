import numpy as np
import matplotlib.pyplot as plt

class statbin:
    def __init__(self, grp_size):
        self.vals = []
        self.grp = np.zeros([2,2])
        self.grp_size = grp_size
        
    def add(self, v):
        self.vals.append(v)
        self.grp_update()

    def grp_update(self):
        if(len(self.vals)%self.grp_size==0):
            self.grp = np.reshape(self.vals, [len(self.vals)/self.grp_size, self.grp_size])
        
    def mean(self):
        return np.mean(self.grp, 1)
        
    def std(self):
        return np.std(self.grp, 1)
       
    def max(self):
        return np.max(self.grp, 1)
    
    def min(self):
        return np.min(self.grp, 1)

    def times(self):
        return np.arange(0, self.grp.shape[0]*self.grp_size, self.grp_size)
 
    def plot(self, lbl=""):
        plt.plot( self.times(), self.min(), color='green', label="Min %s"%(lbl)     )
        plt.plot( self.times(), self.max(), color='red', label="Max %s"%(lbl)       )
        plt.plot( self.times(), self.mean(), color='black', label="Mean %s"%(lbl)   )
        plt.fill_between( self.times(), self.mean()-self.std(), self.mean()+self.std(), facecolor='lightblue', label='std', alpha=0.5)

    def plot2(self, fill_col="lightblue", alpha=0.5, **kwargs):
        plt.plot( self.times(), self.mean(), **kwargs)
        plt.fill_between( self.times(), self.mean()-self.std(), self.mean()+self.std(), facecolor=fill_col, label='std', alpha=alpha)
