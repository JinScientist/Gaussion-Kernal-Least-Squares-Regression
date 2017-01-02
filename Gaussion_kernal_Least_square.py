"""
===================================================================
Support Vector Regression (SVR) and Least Squares with Guassion Kernal
===================================================================
Coder:Jin Huang

"""
print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
M=20 # number of kernal function
lamda=1  #regularization coefficient
############# Linear Combination of Gaussion(LGK) kernal
def LGK(M,x,y,lamda):
	N=len(x)
	mu=np.linspace(0, 6, M)
	phi=np.zeros(shape=[N,M])
	for i in range(1,M):
		phi[:,i]=np.exp(-np.square(x-mu[i-1])/2).reshape(40,)
	phiinv=np.linalg.pinv(phi)
	w=phiinv.dot(y)
	phiT=np.transpose(phi)
	wr=np.linalg.inv(lamda*np.identity(M)+phiT.dot(phi)).dot(phiT).dot(y)
	print('Dimension of Moore-Penrose pseudo-inverse:')
	print(phiinv.shape)
	print('Dimension of y:')
	print(y.shape)
	return(w,wr)#wr:regularized w

###########predict from trained LGK
def LGKpredict(M,w,x):
	N=len(x)	
	phi=np.zeros(shape=[N,M])
	mu=np.linspace(0, 6, M)
	for i in range(1,M):
		phi[:,i]=np.exp(-np.square(x-mu[i-1])/2).reshape(40,)
	ypredict=phi.dot(w.reshape(M,1))
	return(ypredict)
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model

(w,wr)=LGK(M,X,y,lamda)
y_LGK=LGKpredict(M,w,X)
y_LGK_r=LGKpredict(M,wr,X)
np.savetxt("w.csv", w, delimiter=",")
print('Dimension of W_ML:')
print(w.shape)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
lw = 1
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.plot(X, y_LGK, color='maroon', lw=lw, label='Gausian kernal \n Least square model')
plt.plot(X, y_LGK_r, color='lime', lw=lw, label='Regularized Gausian kernal \n Least square model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Gausian Kernal Least Square and Support Vector Regression')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.savefig('SVR.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
