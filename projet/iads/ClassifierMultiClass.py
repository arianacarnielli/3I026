# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:38:30 2019

@author: arian
"""
import numpy as np

# importation de LabeledSet et Classifiers
from . import LabeledSet as ls
from . import Classifiers as cl

class ClassifierMultiClass(cl.Classifier):
    """ Classifieur avec plusieurs classes
    """
    def __init__(self,input_dimension, learning_rate, classClassif, classes):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
                - classClassif : classe de classifieur à utiliser
                - classes : liste des classes à classifier.
            Hypothèse : input_dimension > 0
        """
        self.classes = classes
        self.N = len(classes)
        self.classifiers = [classClassif(input_dimension, learning_rate) for i in range(self.N)]

    def predict(self,x):
        """ rend la prediction sur x 
        """
        iMax = 0
        predMax = -float("inf")
        for i in range(self.N):
            pred = self.classifiers[i].predict(x)
            if pred > predMax:
                predMax = pred
                iMax = i
        return iMax, self.classes[iMax]

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        Les colonnes de l'attribut y de labeledSet doivent correspondre aux
        classes de self.classes
        """
        labSet = ls.LabeledSet(labeledSet.getInputDimension())
        labSet.nb_examples = labeledSet.size()
        labSet.x = labeledSet.x
        for i in range(self.N):
            labSet.y = (labeledSet.y[:, i]).reshape((-1, 1))
            self.classifiers[i].train(labSet)
            
    def accuracy(self, dataset):
        """ Permet de calculer la qualité du système 
        """
        ok = 0
        taille_data = dataset.size()
        for i in range(taille_data):
            cl, _ = self.predict(dataset.getX(i))
            if dataset.y[i, cl] > 0:
            #On dit qu'une prédiction est correcte si la classe prédite est
            #bien une des classes de cette donnée.
                ok+=1
        return (ok/taille_data)*100
        #On renvoie un pourcentage (donc le rapport multiplié par 100)