# Imports

import math as ma
import numpy as np
import matplotlib.pyplot as plt

# Discrétisation

A=0
B=500
N=101 

# Nombre de points de discrétisation

Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N)
discretization = discretization_indexes*Delta

# Paramètres du modèle

mu=-5
a = 50
sigma2 = 12
observation_indexes = [0,20,40,60,80,100]
depth = np.array([0,-4,-12.8,-1,-6.5,0])
unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))

# Questions

## Question 1

def c(h,a = 50,sigma2 = 12):

	if isinstance(h,float) or isinstance(h,int):
		return sigma2*ma.exp(-abs(h)/a)
	else:
		M = []
		for ligne in h:
			L = []
			for coef in ligne:
				L.append(c(coef,a,sigma2))
			M.append(L)
		return np.array(M)
	#else:
	#	print("arguments non valables")

## Question 2

Dist = []
for i in range(101):
	xi = i*500/100
	L = []
	for j in range(101):
		xj = j*500/100
		L.append(abs(xj-xi))
	Dist.append(L)

## Question 3

C = c(Dist)

## Question 4

M1 = []
M2 = []
M3 = []

for i in range(len(C)):
	if i%20 == 0:
		L1 = []
	else :
		L2 = []
		L3 = []
	for j in range(len(C[0])):
		if i%20 == 0 and j%20 == 0:
			L1.append(C[i][j])
		elif i%20 != 0 and j%20 == 0:
			L2.append(C[i][j])
		elif i%20 != 0 and j%20 != 0:
			L3.append(C[i][j])
	if i%20 == 0:
		M1.append(L1)
	else :
		M2.append(L2)
		M3.append(L3)

M1, M2, M3 = np.array(M1), np.array(M2), np.array(M3)

## Question 5 & 6

M1_inv = np.linalg.inv(M1)
M_utile = np.dot(M2,M1_inv)
mu_vec = np.array([-5 for _ in range(101-6)])
mu_vec2 = np.array([depth[i] + 5 for i in range(6)])

esperance = mu_vec + np.dot(M_utile,mu_vec2.T)

# plt.figure()
# plt.plot(unknown_indexes,esperance)
# plt.plot(observation_indexes,depth,linestyle = ' ',marker = '.', label = "observation")
# plt.legend(loc=0)
# plt.title("Esperance conditionnelle en fonction de la position")
# plt.xlabel("position")
# plt.ylabel("Esperance conditionnelle")
# plt.show()

variance = M3 - np.dot(M_utile,M2.T)

# diag = [variance[i][i] for i in range(len(variance))]
# plt.figure()
# plt.plot(unknown_indexes, diag)
# plt.title("Variance conditionnelle en fonction de la position")
# plt.xlabel("position")
# plt.ylabel("Variance conditionnelle")
# plt.show()

## Question 7 


