# Gaussion-Kernal-Least-Squares-Regression
Regularized linear basic function model with Guassion kernal; A 1D least square regression comparing to SVM using linear, polynomial and RBF kernal.

M is the number of Gaussion kernal functions and lamda is the regularization coefficent,Mu is the array of kernal functions' means( Locations of the basis functions in input space).

The the kernal means are chosen equaly distributed in the data space, and spacial scale of kernal funtion is set to one. 

### Below is generated plot from the python code [Gaussion_kernal_Least_square.py](https://github.com/JinScientist/Gaussion-Kernal-Least-Squares-Regression/blob/master/Gaussion_kernal_Least_square.py) with M=20 and lamda=1:

![plot](./regression_plot.png)
There are overfitting when chosing large number of M for unregularized least squares with Gaussion basis function
