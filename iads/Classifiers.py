# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
class Classifier:
    """ Classe pour représenter un classifieur
        Attention: cette classe est une classe abstraite, elle ne peut pas être
        instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        raise NotImplementedError("Please Implement this method")

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        
        raise NotImplementedError("Please Implement this method")
    
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        ok = 0
        taille_data = dataset.size()
        for i in range(taille_data):
            if (dataset.getY(i) * self.predict(dataset.getX(i))) > 0:
            #On dit qu'une prédiction est correcte si elle a le même signe que 
            #la valeur correspondante à cette donnée dans le dataset.
                ok+=1
        return (ok/taille_data)*100
        #On renvoie un pourcentage (donc le rapport multiplié par 100)

# ---------------------------
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        #chaque composante de w est choisi entre -1 et 1
        self.w = 2*np.random.random(input_dimension) - 1
        self.w = self.w / np.linalg.norm(self.w)
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        norm_x = x / np.linalg.norm(x)
        return np.dot(self.w, norm_x)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("Pas d'apprentissage pour ce classifieur")
    
# ---------------------------
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """
 
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.k = k
        
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        taille = self.trainData.size()
        #calcule le tableau de distances
        tab_dist = np.zeros(taille)
        for i in range(taille):
            tab_dist[i] = np.linalg.norm(self.trainData.getX(i) - x)
        #selection des k plus proches voisins
        ind_mins = np.argsort(tab_dist)[:self.k]
        #calcule du score
        cpt_pos = 0
        cpt_min = 0
        for i in ind_mins:
            if self.trainData.getY(i) > 0:
                cpt_pos+=1
            else:
                cpt_min+=1
        return (cpt_pos - cpt_min)/(cpt_pos + cpt_min)

    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        self.trainData = labeledSet

# ---------------------------
