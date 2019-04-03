# -*- coding: utf-8 -*-

"""
Package: iads
Fichier: Classifiers.py
Année: semestre 2 - 2018-2019, Sorbonne Université
"""

# Import de packages externes
import numpy as np
import pandas as pd
import math
import random
import graphviz as gv

import LabeledSet as ls

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
        
# =============================================================================
# Arbres de decision
# =============================================================================

def shannon(P):
    if len(P) <= 1:
        return 0.0
    entropie = 0
    taille = len(P)
    for pi in P:
        if pi != 0:
            #print(pi)
            entropie = entropie - (pi * math.log(pi, taille))
    return entropie  

def discretise2(LSet, col):
    # Extraction de la colonne qui nous intéresse de x
    # et transformation de y en un tableau 1D.
    x = LSet.x[:, col]
    y = LSet.y[:, 0]
    
    # Tri de x par valeurs croissantes et tri correspondant dans y.
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]
    
    # Sélection des indices où y change entre -1 et 1
    indices = np.where(y[:-1] != y[1:])[0]
    
    # Calcul des seuils correspondants
    seuil = (x[indices] + x[indices+1])/2
    
    # Tableau pour garder l'entropie
    entropie = np.zeros(seuil.size)
    for i in range(seuil.size):
        # Ici, on pourrait définir
        # j = indices[i]
        # séparer le tableau y entre y[:(j+1)] et y[(j+1):],
        # calculer l'entropie des deux sous-tableaux et ensuite l'entropie
        # moyenne. Cependant, cela ne marche pas dans un cas :
        # si x[j] == x[j+1], alors ce seuil n'est pas un vrai seuil, on ne
        # peut pas séparer x[j] de x[j+1] par une inégalité.
        # Pour cela, on choisit de re-faire la séparation en comparant x à
        # seuil[i]. On risque ainsi que l'un de nos tableaux soit vide,
        # il faut en tenir compte.
        s = seuil[i]
        yGauche = y[x <= s]
        yDroite = y[x > s]
        
        # Calcul de l'entropie du sous-tableau gauche. Vaut 0 si vide.
        if yGauche.size == 0:
            entGauche = 0.0
        else:
            p = (yGauche==1).sum()/yGauche.size
            entGauche = shannon([p, 1 - p])
            
        # Calcul de l'entropie du sous-tableau droit. Vaut 0 si vide.
        if yDroite.size == 0:
            entDroite = 0.0
        else:
            p = (yDroite==1).sum()/yDroite.size
            entDroite = shannon([p, 1 - p])
            
        entropie[i] = (entGauche*yGauche.size + entDroite*yDroite.size)/y.size
        
    # À la fin, on prend la valeur minimale de l'entropie.
    imin = np.argmin(entropie)
    return (seuil[imin], entropie[imin])

def classe_majoritaire(LabeledSet):
    nb_plus = (LabeledSet.y==1).sum()
    nb_moins = (LabeledSet.y==-1).sum()
    if nb_plus >= nb_moins:
        return 1
    return -1

def entropie(LabeledSet):
    dico_type = {}
    taille_label = LabeledSet.size()
    for y in range(taille_label):
        type = LabeledSet.getY(y)
        if type[0] in dico_type:
            dico_type[type[0]] += 1
        else:
            dico_type[type[0]] = 1
    P = []
    for type in dico_type:
        P.append(dico_type[type]/taille_label)
    #print(P)
    return shannon(P)


def divise(LS, att, seuil):
    Lb1 = ls.LabeledSet(LS.getInputDimension())
    Lb2 = ls.LabeledSet(LS.getInputDimension())
    
    for i in range(LS.size()):
        if LS.getX(i)[att] <= seuil:
            Lb1.addExample(LS.getX(i), LS.getY(i))
        else:
            Lb2.addExample(LS.getX(i), LS.getY(i))
        
    return (Lb1, Lb2)

def construit_AD(LSet, epsilon, level = 0, maxLevel = None):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
    """
    if (entropie(LSet) <= epsilon) or ((maxLevel is not None) and (level >= maxLevel)):
        feuille = ArbreBinaire()
        feuille.ajoute_feuille(classe_majoritaire(LSet))
        return feuille
    taille = LSet.getInputDimension()
    entro = 1.1     
    seuil = None
    att = None
    for col in range(taille): 
        se_test, ent_test = discretise2(LSet, col)
        if entro > ent_test:
            att = col
            entro = ent_test
            seuil = se_test
    if (entropie(LSet) - entro) <= epsilon:
        feuille = ArbreBinaire()
        feuille.ajoute_feuille(classe_majoritaire(LSet))
        return feuille
    noeud = ArbreBinaire()
    LSGauche, LSDroite = divise(LSet, att, seuil)
    ADGauche = construit_AD(LSGauche, epsilon, level = level + 1, maxLevel = maxLevel)
    ADDroite = construit_AD(LSDroite, epsilon, level = level + 1, maxLevel = maxLevel)
    noeud.ajoute_fils(ADGauche, ADDroite, att, seuil)
    return noeud

class ArbreBinaire:
    def __init__(self):
        self.attribut = None   # numéro de l'attribut
        self.seuil = None
        self.inferieur = None # ArbreBinaire Gauche (valeurs <= au seuil)
        self.superieur = None # ArbreBinaire Gauche (valeurs > au seuil)
        self.classe = None # Classe si c'est une feuille: -1 ou +1
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille """
        return self.seuil == None
    
    def ajoute_fils(self,ABinf,ABsup,att,seuil):
        """ ABinf, ABsup: 2 arbres binaires
            att: numéro d'attribut
            seuil: valeur de seuil
        """
        self.attribut = att
        self.seuil = seuil
        self.inferieur = ABinf
        self.superieur = ABsup
    
    def ajoute_feuille(self,classe):
        """ classe: -1 ou + 1
        """
        self.classe = classe
        
    def classifie(self,exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple: +1 ou -1
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] <= self.seuil:
            return self.inferieur.classifie(exemple)
        return self.superieur.classifie(exemple)
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir
            l'afficher
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.attribut))
            self.inferieur.to_graph(g,prefixe+"g")
            self.superieur.to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))
        
        return g
    
    
class ArbreDecision(Classifier):
    # Constructeur
    def __init__(self,epsilon, maxLevel = None):
        # valeur seuil d'entropie pour arrêter la construction
        self.epsilon= epsilon
        self.racine = None
        self.maxLevel = maxLevel
    
    # Permet de calculer la prediction sur x => renvoie un score
    def predict(self,x):
        # classification de l'exemple x avec l'arbre de décision
        # on rend -1 (classe -1) ou 1 (classe 1)
        classe = self.racine.classifie(x)
        if (classe == 1):
            return(1)
        else:
            return(-1)
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD(set,self.epsilon,maxLevel = self.maxLevel)

    # Permet d'afficher l'arbre
    def plot(self):
        gtree = gv.Digraph(format='png')
        return self.racine.to_graph(gtree)
    
def tirage(Vec_X, m, r):
    if r == True:
        res = []
        for i in range(m):
            res.append(random.choice(Vec_X))
        return res
    else:
        return random.sample(Vec_X,m)  
    
def echantillonLS(LS_X, m, r):
    index = tirage([i for i in range (LS_X.size())], m, r)
    res = ls.LabeledSet(LS_X.getInputDimension())
    for ind in index:
        res.addExample(LS_X.getX(ind), LS_X.getY(ind))
    return res

class ClassifierBaggingTree(Classifier):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
    """
    def __init__(self, B, pourc, seuil, r, maxLevel = None):
        self.nb_arbres = B
        self.pourcent = pourc
        self.seuil = seuil
        self.remise = r
        self.maxLevel = maxLevel
    
    def predict(self, x):
        """rend la prediction sur x (-1 ou +1)
        """
        res = 0
        for arbre in self.arbres:
            res += arbre.predict(x)
        if res >= 0:
            return 1
        return -1
        
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = set()
        taille = int(labeledSet.size() * self.pourcent)
        for i in range(self.nb_arbres):
            temp_ls = echantillonLS(labeledSet, taille, self.remise)
            temp_ad= ArbreDecision(self.seuil, maxLevel = self.maxLevel)
            temp_ad.train(temp_ls)   
            self.arbres.add(temp_ad)
            
            
def subSetClasse(labeledSet):
    """
    Separe le LabeledSet pris en argument en deux, suivant les classes 1 et -1.
    """
    res_plus = ls.LabeledSet(labeledSet.getInputDimension())
    res_moins = ls.LabeledSet(labeledSet.getInputDimension())
    for i in range(labeledSet.size()):
        if labeledSet.getY(i) == 1:
            res_plus.addExample(labeledSet.getX(i), 1)
        else:
            res_moins.addExample(labeledSet.getX(i), -1)   
    return res_plus, res_moins

def subSetClasseAmeliore(labeledSet):
    """
    Separe le LabeledSet pris en argument en deux, suivant les classes 1 et -1.
    Utilise des vues des tabelaux X et Y du labeledSet.
    """
    res_plus = ls.LabeledSet(labeledSet.getInputDimension())
    res_moins = ls.LabeledSet(labeledSet.getInputDimension())
    ind_plus = np.where(labeledSet.y == 1)[0]
    ind_moins = np.where(labeledSet.y == -1)[0]
    res_plus.x = labeledSet.x[ind_plus, :]
    res_plus.y = labeledSet.y[ind_plus, :]
    res_moins.x = labeledSet.x[ind_moins, :]
    res_moins.y = labeledSet.y[ind_moins, :]
    res_plus.nb_examples = ind_plus.size
    res_moins.nb_examples = ind_moins.size
    return res_plus, res_moins
    
def fusionSet(LS1, LS2):
    """
    Fusionne les deux LabeledSet LS1 et LS2.
    Les deux doivent avoir le même InputDimension.    
    """
    res = ls.LabeledSet(LS1.getInputDimension())
    for i in range(LS1.size()):
        res.addExample(LS1.getX(i), LS1.getY(i)) 
    for i in range(LS2.size()):
        res.addExample(LS2.getX(i), LS2.getY(i))
    return res

def fusionSetAmeliore(LS1, LS2):
    """
    Fusionne les deux LabeledSet LS1 et LS2.
    Les deux doivent avoir le même InputDimension.    
    """
    res = ls.LabeledSet(LS1.getInputDimension())
    res.x = np.empty((LS1.size() + LS2.size(), LS1.getInputDimension()), dtype = LS1.x.dtype)
    res.y = np.empty((LS1.size() + LS2.size(), 1), dtype = LS1.y.dtype)
    res.nb_examples = res.x.shape[0]
    res.x[:LS1.size(), :] = LS1.x
    res.x[LS1.size():, :] = LS2.x
    res.y[:LS1.size(), :] = LS1.y
    res.y[LS1.size():, :] = LS2.y
    return res

def subSet(labeledSet, x, y):
    """
    Separe le LabeledSet passé en argument en deux :
    - Le premier avec les données entre les indices x et y du LabeledSet original
    - Le deuxième avec les autres données.
    L'indice x est inclus et y est exclu.
    """
    res_xy = ls.LabeledSet(labeledSet.getInputDimension())
    res = ls.LabeledSet(labeledSet.getInputDimension())
    for i in range (labeledSet.size()):
        if i >= x and i < y:
            res_xy.addExample(labeledSet.getX(i), labeledSet.getY(i))
        else:
            res.addExample(labeledSet.getX(i), labeledSet.getY(i))
    return res_xy, res

def subSetAmeliore(labeledSet, x, y):
    """
    Separe le LabeledSet passé en argument en deux :
    - Le premier avec les données entre les indices x et y du LabeledSet original
    - Le deuxième avec les autres données.
    L'indice x est inclus et y est exclu.
    """
    res_xy = ls.LabeledSet(labeledSet.getInputDimension())
    res = ls.LabeledSet(labeledSet.getInputDimension())
    res_xy.x = labeledSet.x[x:y, :]
    res_xy.y = labeledSet.y[x:y, :]
    res.x = np.concatenate((labeledSet.x[:x, :], labeledSet.x[y:, :]))
    res.y = np.concatenate((labeledSet.y[:x, :], labeledSet.y[y:, :]))
    res_xy.nb_examples = res_xy.x.shape[0]
    res.nb_examples = res.x.shape[0]
    return res_xy, res

def echantillonDepuisIndices(LS_X, listIndices):
    #index = tirage([i for i in range (LS_X.size())], m, r)
    res = ls.LabeledSet(LS_X.getInputDimension())
    for ind in listIndices:
        res.addExample(LS_X.getX(ind), LS_X.getY(ind))
    return res

class ClassifierBaggingTreeOOB(ClassifierBaggingTree):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
    """
        
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = dict()
        
        taille = int(labeledSet.size() * self.pourcent)
        for i in range(self.nb_arbres):
            index = tirage([i for i in range (labeledSet.size())], taille, self.remise)
            temp_ls = echantillonDepuisIndices(labeledSet, index)
            temp_ad = ArbreDecision(self.seuil, maxLevel = self.maxLevel)
            temp_ad.train(temp_ls)   
            self.arbres[temp_ad] = index
            
    def accuracyOOB(self, labeledSet):
        """
        Accuracy par la méthode OOB. Il faut que labeledSet soit le même utilisé dans train.
        """
        tBar = 0
        for arbre in self.arbres:
            index = self.arbres[arbre]
            ti = 0
            Ti = np.setdiff1d(np.arange(labeledSet.size()), index)
            for i in Ti:
                if self.predict(labeledSet.getX(i)) == labeledSet.getY(i):
                    ti += 1
            ti /= Ti.size
            tBar += ti
        tBar /= self.nb_arbres
        return tBar*100
        
def construit_AD_aleatoire(LSet, epsilon, nbatt):
    """ LSet : LabeledSet
        epsilon : seuil d'entropie pour le critère d'arrêt 
        nbatt : nombre d'attributs choisis à chaque niveau de la construction. Il faut que nbatt << LSet.getInputDimension().
    """
    if entropie(LSet) <= epsilon:
        feuille = ArbreBinaire()
        feuille.ajoute_feuille(classe_majoritaire(LSet))
        return feuille
    taille = LSet.getInputDimension()
    entro = 1.1     
    seuil = None
    att = None
    cols = tirage(range(taille), nbatt, False)
    for col in cols: 
        se_test, ent_test = discretise2(LSet, col)
        if entro > ent_test:
            att = col
            entro = ent_test
            seuil = se_test
    if (entropie(LSet) - entro) <= epsilon:
        feuille = ArbreBinaire()
        feuille.ajoute_feuille(classe_majoritaire(LSet))
        return feuille
    noeud = ArbreBinaire()
    LSGauche, LSDroite = divise(LSet, att, seuil)   
    ADGauche = construit_AD_aleatoire(LSGauche, epsilon, nbatt)
    ADDroite = construit_AD_aleatoire(LSDroite, epsilon, nbatt)
    noeud.ajoute_fils(ADGauche, ADDroite, att, seuil)
    return noeud


class ArbreDecisionAleatoire(ArbreDecision):
    # Constructeur
    def __init__(self,epsilon,nbatt):
        super(ArbreDecisionAleatoire, self).__init__(epsilon)
        self.nbatt = nbatt
    
    # Permet d'entrainer le modele sur un ensemble de données
    def train(self,set):
        # construction de l'arbre de décision 
        self.set=set
        self.racine = construit_AD_aleatoire(set,self.epsilon, self.nbatt)
        
class ClassifierRandomForest(ClassifierBaggingTree):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
        - Le nombre de colonnes nbatt à utiliser à chaque niveau de l'arbre.
    """
    def __init__(self, B, pourc, seuil, r, nbatt):
        super(ClassifierRandomForest, self).__init__(B, pourc, seuil, r)
        self.nbatt = nbatt
    
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = set()
        taille = int(labeledSet.size() * self.pourcent)
        for i in range(self.nb_arbres):
            temp_ls = echantillonLS(labeledSet, taille, self.remise)
            temp_ad= ArbreDecisionAleatoire(self.seuil, self.nbatt)
            temp_ad.train(temp_ls)
            self.arbres.add(temp_ad)
            
class ClassifierRandomForestOOB(ClassifierRandomForest):
    """Arguments:
        - Le nombre B d'arbres à construire
        - Le pourcentage d'exemples de la base d'apprentissage utilisés pour constituer un échantillon
        - La valeur de seuil d'entropie pour arrêter la construction de chaque arbre
        - Un booléen qui précise si un échantillon est tiré avec ou sans remise
        - Le nombre de colonnes nbatt à utiliser à chaque niveau de l'arbre.
    """
 
    def train(self, labeledSet):
        """ Permet d'entrainer le modele sur l'ensemble donné
        """
        self.arbres = dict()
        taille = int(labeledSet.size() * self.pourcent)
        for i in range(self.nb_arbres):
            index = tirage([i for i in range (labeledSet.size())], taille, self.remise)
            temp_ls = echantillonDepuisIndices(labeledSet, index)
            temp_ad= ArbreDecisionAleatoire(self.seuil, self.nbatt)
            temp_ad.train(temp_ls)
            self.arbres[temp_ad] = index
            
    def accuracyOOB(self, labeledSet):
        """
        Accuracy par la méthode OOB. Il faut que labeledSet soit le même utilisé dans train.
        """
        tBar = 0
        for arbre in self.arbres:
            index = self.arbres[arbre]
            ti = 0
            Ti = np.setdiff1d(np.arange(labeledSet.size()), index)
            for i in Ti:
                if self.predict(labeledSet.getX(i)) == labeledSet.getY(i):
                    ti += 1
            ti /= Ti.size
            tBar += ti
        tBar /= self.nb_arbres
        return tBar*100
