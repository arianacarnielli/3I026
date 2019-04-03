# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: kmoyennes.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# ---------------------------
# Fonctions pour les k-moyennes

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt

import math
import random

import numpy as np

# ---------------------------
# Dans ce qui suit, remplacer la ligne "raise.." par les instructions Python
# demandées.
# ---------------------------

# Normalisation des données :

# ************************* Recopier ici la fonction normalisation()
def normalisation(df):
    """ DataFrame -> DataFrame
        rend le dataframe obtenu par normalisation des données selon 
             la méthode vue en cours 8.
    """
    return (df - df.min())/(df.max() - df.min())
# -------
# Fonctions distances

# ************************* Recopier ici la fonction dist_vect()
def dist_vect(v1, v2):
    """ Series**2 -> float
        rend la valeur de la distance euclidienne entre les 2 vecteurs
    """
    return np.linalg.norm(v1 - v2)

# -------
# Calculs de centroïdes :
# ************************* Recopier ici la fonction centroide()
def centroide(df):
    """ DataFrame -> DataFrame
        Hypothèse: len(M) > 0
        rend le centroïde des exemples contenus dans M
    """
    return pd.DataFrame(df.mean()).transpose()

# -------
# Inertie des clusters :
# ************************* Recopier ici la fonction inertie_cluster()
def inertie_cluster(df):
    """ DataFrame -> float
        DF: DataFrame qui représente un cluster
        L'inertie est la somme (au carré) des distances des points au centroide.
    """
    centre = centroide(df)
    inertie = 0
    for i, row in df.iterrows():
        inertie += dist_vect(row, centre)**2
    return inertie 


# -------
# Algorithmes des K-means :
# ************************* Recopier ici la fonction initialisation()
def initialisation(k,df):
    """ int * DataFrame -> DataFrame
        K : entier >1 et <=n (le nombre d'exemples de DF)
        DF: DataFrame contenant n exemples
    """
    index = random.sample(range(df.shape[0]), k)
    res = df.iloc[index]
    return res


# -------
# ************************* Recopier ici la fonction plus_proche()
def plus_proche(exemple,df_centre):
    """ Series * DataFrame -> int
        Exe : Series contenant un exemple
        Centres : DataFrame contenant les K centres
    """
    dist_min = float("inf")
    index_centre = 0
    index = 0
    for _, row in df_centre.iterrows():
        dist = dist_vect(exemple, row)
        if dist < dist_min:
            dist_min = dist
            index_centre = index
        index += 1
    return index_centre

# -------
# ************************* Recopier ici la fonction affecte_cluster()
def affecte_cluster(df_learn, df_centre):
    """ DataFrame * DataFrame -> dict[int,list[int]]
        Base: DataFrame contenant la base d'apprentissage
        Centres : DataFrame contenant des centroides
    """
    dico = {}
    for i in range(len(df_learn)):
        index_centre = plus_proche(df_learn.iloc[i],df_centre)
        if index_centre in dico:
            dico[index_centre].append(i)
        else:
            dico[index_centre] = [i]
    return dico

# -------
# ************************* Recopier ici la fonction nouveaux_centroides()
def nouveaux_centroides(df_learn, dico_aff):
    """ DataFrame * dict[int,list[int]] -> DataFrame
        Base : DataFrame contenant la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    df_res = pd.DataFrame()
    for centre in dico_aff:
        n_centre = df_learn.iloc[dico_aff[centre]]
        n_centre = n_centre.mean()
        df_res = df_res.append(n_centre, ignore_index = True)
    return df_res

# -------
# ************************* Recopier ici la fonction inertie_globale()
def inertie_globale(df_learn, dico_aff):
    """ DataFrame * dict[int,list[int]] -> float
        Base : DataFrame pour la base d'apprentissage
        U : Dictionnaire d'affectation
    """
    inertie = 0
    for centre in dico_aff:
        df_cluster = df_learn.iloc[dico_aff[centre]]
        inertie += inertie_cluster(df_cluster)
    return inertie
# -------
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes(k, df_learn, epsilon, iter_max, verbose = True):
    """ int * DataFrame * float * int -> tuple(DataFrame, dict[int,list[int]])
        K : entier > 1 (nombre de clusters)
        Base : DataFrame pour la base d'apprentissage
        epsilon : réel >0
        iter_max : entier >1
    """
    centroides = initialisation(k, df_learn)
    dico_aff = affecte_cluster(df_learn, centroides)
    centroides = nouveaux_centroides(df_learn, dico_aff)
    inertie = inertie_globale(df_learn, dico_aff)
    
    for iter in range(iter_max):
        dico_aff = affecte_cluster(df_learn, centroides)
        centroides = nouveaux_centroides(df_learn, dico_aff)
        n_inertie = inertie_globale(df_learn, dico_aff)
        if verbose == True:
            print("iteration ", iter, " Inertie : ", n_inertie, " Difference: ",  abs(n_inertie - inertie))
        if abs(n_inertie - inertie) < epsilon:
            break
        else:
            inertie = n_inertie 
    return centroides, dico_aff
# -------
# Affichage :
# ************************* Recopier ici la fonction affiche_resultat()
def affiche_resultat(df_learn, df_centre, dico_aff):
    """ DataFrame **2 * dict[int,list[int]] -> None
    """    
    # Remarque: pour les couleurs d'affichage des points, quelques exemples:
    # couleurs =['darkviolet', 'darkgreen', 'orange', 'deeppink', 'slateblue', 'orangered','y', 'g', 'b']
    # voir aussi (google): noms des couleurs dans matplolib
    for i, centre in enumerate(dico_aff):
        df = df_learn.iloc[dico_aff[centre]]
        plt.scatter(df['X'],df['Y'], color=("C" + str(i % 10)))
    plt.scatter(df_centre['X'],df_centre['Y'],color='r',marker='x')
# -------