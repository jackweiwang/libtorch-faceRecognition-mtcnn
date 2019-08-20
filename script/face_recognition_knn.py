import numpy as np
from sklearn import neighbors
import time
import sys
 
def linerf(x):
    y=-118*x+115
    return y
  
def loadTrainData(path):
	
    frtrain1 = np.genfromtxt(path, delimiter=' ', dtype=str)

    Xtrain = frtrain1[:, 1:129]
    Ytrain = frtrain1[:, 0:1].ravel()
    #Ytrain = frtrain1[:, 128:129].ravel()
    return Xtrain,Ytrain
	
def loadTestData(path):
    frtest1 = np.genfromtxt(path, delimiter=' ', dtype=str)
    #Xtest = frtest1[:, 0:128].ravel()
    return frtest1
 

	
#加载训练集

Xtrain, Ytrain=loadTrainData(sys.argv[1])

loadtime1=time.time()
clf = neighbors.KNeighborsClassifier(algorithm="ball_tree", metric='euclidean', n_neighbors=1)
clf.fit(Xtrain, Ytrain)
loadtime2=time.time()
print("train time",loadtime2-loadtime1)

def main():

	t1=time.time()
	Xtest = loadTestData(sys.argv[2])

	tzlist=[]
	for i in Xtest:
		tzlist.append(i)
	resname=clf.predict([Xtest])
	distance,ind=clf.kneighbors([Xtest],n_neighbors=1,return_distance=True)
	
	score=linerf(distance)
	
	score=min(98,score)
	if distance[0][0]>0.4:
		resname[0]='unknown'
	print(resname[0])
	
	t2=time.time()
	
	print('test time ',t2-t1)
	
if __name__ == '__main__':
	main()
