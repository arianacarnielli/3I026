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
        
class ClassifierPerceptronRandom(Classifier):
    def __init__(self, input_dimension):
        """ Argument:
                - input_dimension (int) : dimension d'entrée des exemples
            Hypothèse : input_dimension > 0
        """  
        v = np.random.rand(input_dimension)     # vecteur aléatoire à input_dimension dimensions
        self.w = 2*v - 1
        self.w = self.w / np.linalg.norm(self.w)     # on normalise par la norme de w
        
        
        #v = np.random.rand(input_dimension)     # en effet, cette façon de caulculer n'est pas correcte
        #self.w = (2* v - 1) / np.linalg.norm(v)  

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = np.dot(x, self.w)
        return z
        
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """        
        print("No training needed")

# ---------------------------

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(input_dimension)
        self.e = learning_rate

    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        return np.dot(x, self.w)

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        # parcours des données du labeledSet en ordre aléatoire
        ordre = np.arange(labeledSet.size())
        np.random.shuffle(ordre)
        for i in ordre:
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            if z * labeledSet.getY(i) <= 0:
                self.w += self.e * elem * labeledSet.getY(i)
                # La normalisation de w a été choisie pour garantir
                #que chaque modification de w est petite (de l'ordre de self.e) 
                #par rapport à la valeur précédente de w.
                self.w /= np.linalg.norm(self.w)
        
    def bad_train(self,labeledSet):
        """ Entrainement sur l'ensemble donné sans normalisation de self.w (résultats mauvais)
        """
        # parcours des données du labeledSet en ordre aléatoire
        ordre = np.arange(labeledSet.size())
        np.random.shuffle(ordre)
        for i in ordre:
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            if z * labeledSet.getY(i) <= 0:
                self.w += self.e * elem * labeledSet.getY(i)
                # ici on ne normalize pas. Cela cause des alterations très grandes de la
                # valeur de l'accuracy comme bien demontré dans un cellule a suivre  

# ---------------------------   
    
class ClassifierPerceptronKernel(Classifier):
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.w = np.zeros(dimension_kernel)
        self.e = learning_rate
        self.k = kernel

        
    def predict(self,x):
        """ rend la prediction sur x (-1 ou +1)
        """
        z = self.k.transform(x)
        res = np.dot(z, self.w)
        return res

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        ordre = np.arange(labeledSet.size())
        np.random.shuffle(ordre)
        for i in ordre:
            elem = labeledSet.getX(i)
            elem = self.k.transform(elem)
            z = np.dot(elem, self.w)
            if z * labeledSet.getY(i) <= 0:
                self.w += self.e * elem * labeledSet.getY(i)
                # La normalisation de w a été choisie pour garantir
                #que chaque modification de w est petite (de l'ordre de self.e) 
                #par rapport à la valeur précédente de w.
                self.w /= np.linalg.norm(self.w)
        
        
        
    def train_bad(self,labeledSet):
        """ Version sans normalisation de w 
        """  
        i = np.random.randint(labeledSet.size())
        elem = labeledSet.getX(i)
        elem = self.k.transform(elem)
        z = np.dot(elem, self.w)
        if z * labeledSet.getY(i) <= 0:
            self.w = self.w + self.e *(elem * labeledSet.getY(i))
            
# --------------------------- 
            
class ClassifierGradientSto(Classifier):
    """ Descent du gradient stochastique
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.e = learning_rate
        #w initialisé de façon aléatoire
        self.w = (np.random.rand(input_dimension) - 0.5) * self.e
       

    def predict(self,x):
        """ rend la prediction sur x 
        """
        return np.dot(x, self.w)

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        # parcours des données du labeledSet en ordre aléatoire
        ordre = np.arange(labeledSet.size())
        np.random.shuffle(ordre)
        for i in ordre:
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            #pas necessaire de tester, on change w toujours
            self.w += self.e * (labeledSet.getY(i) - z) * elem 
            
    def loss(self, labeledSet):
        """Calcul de la fonction de loss sur le dataset labeledSet.  
        """
        val_loss = 0
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            val_loss += (labeledSet.getY(i) - z)**2
        return val_loss/labeledSet.size()
    
# --------------------------- 
        
class ClassifierGradientBatch(Classifier):
    """ Descent du gradient en batch
    """
    def __init__(self,input_dimension,learning_rate):
        """ Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - learning_rate :
            Hypothèse : input_dimension > 0
        """
        self.e = learning_rate
        #w initialisé de façon aléatoire
        self.w = (np.random.rand(input_dimension) - 0.5) * self.e
       

    def predict(self,x):
        """ rend la prediction sur x 
        """
        return np.dot(x, self.w)

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        # parcours des données du labeledSet
        gradient = np.zeros(self.w.size)
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            gradient += (labeledSet.getY(i) - z) * elem
        self.w += self.e * gradient / labeledSet.size()
        
        
    def loss(self, labeledSet):
        """Calcul de la fonction de loss sur le dataset labeledSet. 
        """
        val_loss = 0
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            val_loss += (labeledSet.getY(i) - z)**2
        return val_loss/labeledSet.size()
    
# ---------------------------
        
class ClassifierGradientStoKernel(Classifier):
    """ Descent du gradient stochastique kernelisé
    """
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - dimension_kernel (int) : dimension du kernel
                - learning_rate : e
            Hypothèse : dimension_kernel > 0
        """
        self.e = learning_rate
        #w initialisé de façon aléatoire
        self.w = (np.random.rand(dimension_kernel) - 0.5) * self.e
        self.k = kernel
       

    def predict(self,x):
        """ rend la prediction sur x 
        """
        z = self.k.transform(x)
        res = np.dot(z, self.w)
        return res

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        # parcours des données du labeledSet en ordre aléatoire
        ordre = np.arange(labeledSet.size())
        np.random.shuffle(ordre)
        for i in ordre:
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            elem = self.k.transform(elem)
            #pas necessaire de tester, on change w toujours
            self.w += self.e * (labeledSet.getY(i) - z) * elem 
            
            
    def loss(self, labeledSet):
        """Calcul de la fonction de loss sur le dataset labeledSet.
        """
        val_loss = 0
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            val_loss += (labeledSet.getY(i) - z)**2
        return val_loss/labeledSet.size()    
    
    
# ---------------------------
        
class ClassifierGradientBatchKernel(Classifier):
    """ Descent du gradient en batch kernelisé
    """
    def __init__(self,dimension_kernel,learning_rate,kernel):
        """ Argument:
                - dimension_kernel (int) : dimension du kernel
                - learning_rate : e
            Hypothèse : dimension_kernel > 0
        """
        self.e = learning_rate
        #w initialisé de façon aléatoire
        self.w = (np.random.rand(dimension_kernel) - 0.5) * self.e
        self.k = kernel
       

    def predict(self,x):
        """ rend la prediction sur x 
        """
        z = self.k.transform(x)
        res = np.dot(z, self.w)
        return res

    
    def train(self,labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        # parcours des données du labeledSet
        gradient = np.zeros(self.w.size)
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            elem = self.k.transform(elem)
            gradient += (labeledSet.getY(i) - z) * elem
        self.w += self.e * gradient / labeledSet.size()
        
        
    def loss(self, labeledSet):
        """Calcul de la fonction de loss sur le dataset labeledSet.
        """
        val_loss = 0
        for i in range(labeledSet.size()):
            elem = labeledSet.getX(i)
            z = self.predict(elem)
            val_loss += (labeledSet.getY(i) - z)**2
        return val_loss/labeledSet.size()
    
    
# ---------------------------

        