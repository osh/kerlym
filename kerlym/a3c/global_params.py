#!/usr/bin/env python
import threading, Queue
import numpy as np
import copy

class global_params(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.weights = None
        self.update_queue = Queue.Queue()
        self.weight_lock = threading.Lock()
        self.finished = False

        self.alpha = 0.99
        self.lr = lambda: 10**np.random.uniform(-2,-4)
    
    def update(self, update):
        self.update_queue.put(update)


    def get_weights(self):
        self.weight_lock.acquire()
        w = copy.copy(self.weights)
        self.weight_lock.release()
        return w
        
    def run(self):
        while not self.finished:
            holding = False
#            try:
            if True:
                u = self.update_queue.get(block=True)
                #u = self.update_queue.get(block=True, timeout=1.0)
                print "GP: Do Update "

                self.weight_lock.acquire()
                holding = True
                wts,grad = u

                # set weights if its empty
                if self.weights == None:
                    self.weights = copy.copy(wts)

                    # initialize momentum with zeros of weight shape ..
                    self.g = copy.copy(self.weights)
                    for g_ in self.g:
                        for g__ in g_:
                            g__ *= 0

                # learning hyperparams...
                lr = self.lr()
                alpha = self.alpha
                epsilon = 1e-3

                # perform grad update with Shared RMSProp
                for netidx in [0,1]:
                    for i in range(0,len(self.g[netidx])):
                        self.g[netidx][i] = alpha*self.g[netidx][i] + (1.0-alpha)*np.power(grad[netidx][i],2)                              # (S2)
                        self.weights[netidx][i] = self.weights[netidx][i] - lr * grad[netidx][i] / np.sqrt(self.g[netidx][i] + epsilon)     # (S3) 

#            except:
#                # timeout condition
#                print "GP: exception"
#                pass
            if holding:
                self.weight_lock.release()

            



