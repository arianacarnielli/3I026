# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: utils.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions utiles pour les TDTME de 3i026

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importation de LabeledSet
from . import LabeledSet as ls

def plot2DSet(set):
    """ LabeledSet -> NoneType
        Hypothèse: set est de dimension 2
        affiche une représentation graphique du LabeledSet
        remarque: l'ordre des labels dans set peut être quelconque
    """
    S_pos = set.x[np.where(set.y == 1),:][0]      # tous les exemples de label +1
    S_neg = set.x[np.where(set.y == -1),:][0]     # tous les exemples de label -1
    plt.scatter(S_pos[:,0],S_pos[:,1],marker='o') # 'o' pour la classe +1
    plt.scatter(S_neg[:,0],S_neg[:,1],marker='x') # 'x' pour la classe -1

def plot_frontiere(set,classifier,step=10):
    """ LabeledSet * Classifier * int -> NoneType
        Remarque: le 3e argument est optionnel et donne la "résolution" du tracé
        affiche la frontière de décision associée au classifieur
    """
    mmax=set.x.max(0)
    mmin=set.x.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    plt.contourf(x1grid,x2grid,res,colors=["red","cyan"],levels=[-1000,0,1000])
    
# ------------------------ 

def createGaussianDataset(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ 
        rend un LabeledSet 2D généré aléatoirement.
        Arguments:
        - positive_center (vecteur taille 2): centre de la gaussienne des points positifs
        - positive_sigma (matrice 2*2): variance de la gaussienne des points positifs
        - negative_center (vecteur taille 2): centre de la gaussienne des points négative
        - negative_sigma (matrice 2*2): variance de la gaussienne des points négative
        - nb_points (int):  nombre de points de chaque classe à générer
    """
    res = ls.LabeledSet(2)
    for i in range(nb_points):    
    #La méthode addExample ne permet l'ajout d'une seule donnée à la fois.
        res.addExample(np.random.multivariate_normal(positive_center, positive_sigma), 1)
        res.addExample(np.random.multivariate_normal(negative_center, negative_sigma), -1)
    
    return res
    
def createXOR(nb_points,var):
    set1 = createGaussianDataset(np.array([1,1]),np.array([[var,0],[0,var]]),np.array([1,-1]),np.array([[var,0],[0,var]]),nb_points)
    set2 = createGaussianDataset(np.array([-1,-1]),np.array([[var,0],[0,var]]),np.array([-1,1]),np.array([[var,0],[0,var]]),nb_points)
    taille = set2.size()
    for i in range(taille):
        set1.addExample(set2.getX(i), set2.getY(i))
    return set1  

# ------------------------ 
    
class KernelBias:
    def transform(self,x):
        y=np.asarray([x[0],x[1],1])
        return y


class KernelPoly:
    def transform(self,x):
        y = np.asarray([1, x[0], x[1], x[0]*x[0], x[1]*x[1], x[0]*x[1]])
        return y
    
    
class KernelPolyMultiD:
    def transform(self,x):
        xi = x.reshape((x.size, 1))
        xj = x.reshape((1, x.size))
        mat = xi.dot(xj)
        y = mat[np.triu_indices(x.size)]
        y = np.concatenate((np.array([1]), x, y))
        return y        
    
# ------------------------

def split(l_set,  p_train = 0.85):
    """
    Sépare le LabeledSet l_set de façon aléatoire en deux LabeledSet, 
    l'un pour l'entrainement et l'autre pour les tests. 
    Le paramètre p_train donne la probabilité qu'une entrée de l_set soit mise dans le LabeledSet d'entrainement.
    """
    l_set_train = ls.LabeledSet(l_set.getInputDimension())
    l_set_test = ls.LabeledSet(l_set.getInputDimension())
    taille = l_set.size()
    for i in range(taille):
        if np.random.rand() < p_train:
            l_set_train.addExample(l_set.getX(i), l_set.getY(i))
        else:
            l_set_test.addExample(l_set.getX(i), l_set.getY(i))
    return l_set_train, l_set_test