#imports
import math as ma
import numpy as np

# Questions

## Question 1

def c(h,a = 50,sigma2 = 12):

	if isinstance(h,int) :
		return sigma2*ma.exp(-abs(h)/a)
	if isinstance(h,np.array) :
		M = np.array([])
		for ligne in h:
			L = []
			for coef in ligne:
				L.append(c(coef,a,sigma2))
			M.append(L)
		return M
	else:
		print("arguments non valables")

## Question 2

Dist = []
for i in range(1,N+1):
	xi = 500/i
	L = []
	for j in range(1,N+1):
		xj = 500/j
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
		L2 = []
	else :
		L3 = []
	for j in range(len(C[0])):
		if i%20 == 0 and j%20 == 0:
			L1.append(C[i][j])
		elif i%20 == 0 and j%20 != 0:
			L2.append(C[i][j])
		else i%20 != 0 and j%20 != 0:
			L3.append(C[i][j])
	if i%20 == 0:
		M1.append(L1)
		M2.append(L2)
	else :
		M3.append(L3)

## Question 5 & 6

M1_inv = np.linalg.inv(M1)
M_utile = np.dot(M2,M1_inv)
mu_vec = np.array([-5 for _ in range(100-6)])
mu_vec2 = np.array([-5 for _ in range(6)])

esperance = mu_vec + np.dot(M_utile,mu_vec2)

variance = M3 - np.dot(M_utile,M2)

## Question 7 


