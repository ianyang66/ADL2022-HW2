import numpy as np
eval_EM = []

test = np.load('qaPertlinearExEpoch10/eval_EM.npy')
#test = np.load('MCmaclargegr8epo2K/eval_accuracy.npy')
# test = np.load('MCmaclargegr8epo2K/eval_accuracy.npy')
#str = ','.join(str(i) for i in test)
#print(str)
#print('occurrence of letter a:', str.count('1.0'))
print(test)
