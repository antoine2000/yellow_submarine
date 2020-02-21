# Imports

import math as ma
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

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

# On calcule d'abord la matrice R, qui est la racine carrée de la variance de Z.

R = scipy.linalg.sqrtm(variance)

# On prend une fonction qui fait une simulation aléatoire d'un vecteur gaussion centré normé

def simulation_Z(N = 95):
	Y = np.random.normal(0,1,N)
	Z = esperance + np.dot(R,Y)
	Z_complete = []
	for i in range(101):
		if i%20 == 0:
			Z_complete.append(depth[i//20])
		else:
			Z_complete.append(Z[i-i//20-1])
	return Z_complete

def simulation_L(N = 95):
	Z = simulation_Z(N)
	delta = 5
	S = 0
	temp = Z[0]
	for z in Z[1:]:
		S += ma.sqrt(delta**2 + np.real((z-temp))**2)
		temp = 1*z
	return S

def affichage_simulations(number = 1, N = 95):
	plt.figure()
	plt.title("Esperance conditionnelle en fonction de la position")
	plt.xlabel("position")
	plt.ylabel("Esperance conditionnelle")
	plt.plot(unknown_indexes,esperance,label = "esperance")
	plt.plot(observation_indexes, depth, label = "observations", marker = "o", linestyle = " ")
	for i in range(number):
		plt.plot(discretization_indexes,simulation_Z(N), label = "simulation numéro " + str(i+1))
	plt.legend(loc = 0)
	plt.show()

def estimation(number, N = 95):
	simulations = np.array([simulation_L(N) for _ in range(95)])
	return np.mean(simulations)

esperance_complete = []
for i in range(101):
		if i%20 == 0:
			esperance_complete.append(depth[i//20])
		else:
			esperance_complete.append(esperance[i-i//20-1])

longueur_esperee = 0
temp = esperance_complete[0]
for e in esperance_complete[1:]:
		longueur_esperee += ma.sqrt(5**2 + (e-temp)**2)
		temp = 1*e


def moyenne_evoluante(number, N = 95):
	simulations = []
	moyennes = []
	for k in range(1,number+1):
		simulations.append(simulation_L(N))
		moyennes.append(sum(simulations)/k)
	return moyennes

def affichage_moyenne_evoluante(number, N = 95):
	numbers = [i for i in range(10,number)]
	moyennes = moyenne_evoluante(number,N)[10:]
	plt.figure()
	plt.title("Evolution de la moyenne des longueurs de cable simulee au cours des simulations")
	plt.xlabel("numero de la simulation")
	plt.ylabel("moyenne depuis le debut des simulation")
	plt.plot(numbers,moyennes)
	plt.plot(numbers,[moyennes[-1] for _ in range(number - 10)],label = "valeur finale")
	plt.plot(numbers,[moyennes[-1]*.995 for _ in range(number - 10)],label = "intervalle à -0,005%")
	plt.plot(numbers,[moyennes[-1]*1.005 for _ in range(number - 10)],label = "intervalle à +0,005%")
	plt.legend(loc=0)
	plt.show()

def hist_moyenne(number, N = 95):
	moyennes = moyenne_evoluante(number,N)
	plt.figure()
	plt.title("Histogramme des longueurs de cable simulee au cours des simulations")
	plt.xlabel("longueur")
	plt.ylabel("nombre de simulations")
	plt.hist(moyennes, bins = np.arange(522.3,522.7,0.001))
	plt.show()		

def proportion_dedans(number, simulations, moyenne, ecart):
	nombre_dedans = 0
	for simulation in simulations:
		if simulation > moyenne - ecart and simulation < moyenne + ecart :
			nombre_dedans += 1
	return nombre_dedans / number

def intervalle_confiance_naïve(number, N = 95):
	simulations_L = np.array([simulation_L(N) for _ in range(number)])
	moyenne = np.mean(simulations_L)
	ecart_moyenne = 0
	while proportion_dedans(number, simulations_L,moyenne,ecart_moyenne) < 0.95 :
		ecart_moyenne += 0.01
	return 2*ecart_moyenne


def intervalle_confiance_approx(number, N = 95):
	simulations_L = np.array([simulation_L(N) for _ in range(number)])
	moyenne = np.mean(simulations_L)
	ecart_type = ma.sqrt(np.mean( (simulations_L - moyenne)**2 ))
	return 3*ecart_type

# résultats des intervalles de confiance

# plt.figure()
# plt.title("écart carateristique de l'intervalle de confiance à 95% de la longueur du câble en fonction de la méthode et du nombre de simulations")
# plt.xlabel("nombre de simulations, en échelle logarithmique")
# plt.xscale("log")
# plt.ylabel("écart carateristique")
# liste_nombre = [100,1_000,10_000,100_000]
# liste_naive = [intervalle_confiance_naïve(n) for n in liste_nombre]
# liste_approx = [intervalle_confiance_approx(n) for n in liste_nombre]
# plt.plot(liste_nombre,liste_naive, label = "méthode naïve", marker = "o", linestyle = " ")
# plt.plot(liste_nombre,liste_approx, label = "méthode par approximation", marker = "o", linestyle = " ")
# plt.legend(loc = 0)
# plt.show()

# Probabilité des 425 mètres

def probabilité_supérieure_525(number, N = 95):
	simulations_L = np.array([simulation_L(N) for _ in range(number)])
	nombre_sup = 0
	for simulation in simulations_L:
		if simulation >= 525:
			nombre_sup += 1
	return nombre_sup / number

print(probabilité_supérieure_525(100))
print(probabilité_supérieure_525(1000))
print(probabilité_supérieure_525(10000))
print(probabilité_supérieure_525(100000))
