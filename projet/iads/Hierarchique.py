# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:56:10 2019

@author: arian
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy

def normalisation(df):
    mat = df.to_numpy() 
    for i in range(mat.shape[1]):
        mat[:,i] = (mat[:,i] - mat[:,i].min())/(mat[:,i].max() - mat[:,i].min())
    return mat

def dist_euclidienne_vect(vect1, vect2):
    return np.linalg.norm(vect1 - vect2)

def dist_manhattan_vect(vect1, vect2):
    return np.linalg.norm((vect1 - vect2), 1)

def dist_vect(chaine, vect1, vect2):
    if chaine == "euclidienne":
        return dist_euclidienne_vect(vect1, vect2)
    else:
        return dist_manhattan_vect(vect1, vect2)
    
def centroide(mat):
    return mat.mean(0)

def dist_groupes(chaine, mat1, mat2):
    cent1 = centroide(mat1)
    cent2 = centroide(mat2) 
    #print("centroide 1 : ", cent1)
    #print("centroide 2 : ", cent2)
    return dist_vect(chaine, cent1, cent2)

def initialise(M):
    return {i: M[i,:].reshape((1,M.shape[1])) for i in range(M.shape[0])}

def fusionne(chaine, C0, verbose = True):
    minA = 0
    minB = 0
    dist_min = float("inf")
    #print("Début de la fusion...")
    for i in C0:
        #print("i =",i)
        for j in C0:
            if i != j:
                if dist_groupes(chaine, C0[i], C0[j]) < dist_min:
                    minA = i
                    minB = j
                    dist_min = dist_groupes(chaine, C0[i], C0[j])
    res = {i: C0[i] for i in C0 if i != minA and i != minB}
    res[max(C0) + 1] = np.concatenate((C0[minA], C0[minB]))
    if verbose:
        print("Fusion de {} et {} pour une distance de {}".format(minA, minB, dist_min))
    return res, minA, minB, dist_min

def clustering_hierarchique_complet(chaine, df, x_names = None, verbose = True):
    courant = initialise(df)       # clustering courant, au départ:s données data_2D normalisées
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        #print(len(courant))
        novo, k1, k2, dist_min = fusionne(chaine, courant, verbose = verbose)
        if(len(M_Fusion) == 0):
            M_Fusion = [k1, k2, dist_min, 2]
        else:
            M_Fusion = np.vstack([M_Fusion,[k1, k2, dist_min, 2]])
        courant = novo
    return M_Fusion, dessine_dendrogramme(M_Fusion, x_names)

def clustering_hierarchique_seuil(chaine, df, dist_seuil, verbose = True):
    courant = initialise(df)       # clustering courant, au départ:s données data_2D normalisées
    clusters = {i: {i} for i in range(df.shape[0])}
    M_Fusion = []                        # initialisation
    while len(courant) >=2:              # tant qu'il y a 2 groupes à fusionner
        #print(len(courant))
        novo, k1, k2, dist_min = fusionne(chaine, courant, verbose = verbose)
        if dist_min > dist_seuil:
            break
        clusters[max(max(clusters), k1, k2)+1] = clusters.pop(k1) | clusters.pop(k2)
        assert clusters.keys() == novo.keys()
        if(len(M_Fusion) == 0):
            M_Fusion = [k1, k2, dist_min, 2]
        else:
            M_Fusion = np.vstack([M_Fusion,[k1, k2, dist_min, 2]])
        courant = novo
    return clusters, courant


def dessine_dendrogramme(M_Fusion, x_names = None):
    # Paramètre de la fenêtre d'affichage: 
    fig, ax = plt.subplots(figsize=(30, 15)) # taille : largeur x hauteur
    ax.set_title('Dendrogramme', fontsize=25)    
    ax.set_xlabel('Exemple', fontsize=25)
    ax.set_ylabel('Distance', fontsize=25)

    # Construction du dendrogramme à partir de la matrice M_Fusion:
    res = scipy.cluster.hierarchy.dendrogram(
        M_Fusion,
        leaf_font_size=18.,  # taille des caractères de l'axe des X
    )
    
    if x_names is not None:
        curr_labels = [int(label.get_text()) for label in ax.get_xticklabels()]
        ax.set_xticklabels([x_names[i] for i in curr_labels])
        
    # Affichage du résultat obtenu:
    plt.show()
 
    return res