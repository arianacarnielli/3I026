# -*- coding: utf-8 -*-

import numpy as np
import math

import sys
sys.path.append('../../')

#import iads as iads
from iads import LabeledSet as ls
#from iads import Classifiers as cl
#from iads import utils as ut

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

def discretise(LSet, col):
    """ LabeledSet * int -> tuple[float, float]
        Hypothèse: LSet.size() >= 2
        col est le numéro de colonne sur X à discrétiser
        rend la valeur de coupure qui minimise l'entropie ainsi que son entropie.
    """
    # initialisation:
    min_entropie = 1.1  # on met à une valeur max car on veut minimiser
    min_seuil = 0.0     
    # trie des valeurs:
    ind= np.argsort(LSet.x,axis=0)
    
    # calcul des distributions des classes pour E1 et E2:
    inf_plus  = 0               # nombre de +1 dans E1
    inf_moins = 0               # nombre de -1 dans E1
    sup_plus  = 0               # nombre de +1 dans E2
    sup_moins = 0               # nombre de -1 dans E2       
    # remarque: au départ on considère que E1 est vide et donc E2 correspond à E. 
    # Ainsi inf_plus et inf_moins valent 0. Il reste à calculer sup_plus et sup_moins 
    # dans E.
    for j in range(0,LSet.size()):
        if (LSet.getY(j) == -1):
            sup_moins += 1
        else:
            sup_plus += 1
    nb_total = (sup_plus + sup_moins) # nombre d'exemples total dans E
    
    # parcours pour trouver le meilleur seuil:
    for i in range(len(LSet.x)-1):
        v_ind_i = ind[i]   # vecteur d'indices
        courant = LSet.getX(v_ind_i[col])[col]
        lookahead = LSet.getX(ind[i+1][col])[col]
        val_seuil = (courant + lookahead) / 2.0;
        # M-A-J de la distrib. des classes:
        # pour réduire les traitements: on retire un exemple de E2 et on le place
        # dans E1, c'est ainsi que l'on déplace donc le seuil de coupure.
        if LSet.getY(ind[i][col])[0] == -1:
            inf_moins += 1
            sup_moins -= 1
        else:
            inf_plus += 1
            sup_plus -= 1
        # calcul de la distribution des classes de chaque côté du seuil:
        nb_inf = (inf_moins + inf_plus)*1.0     # rem: on en fait un float pour éviter
        nb_sup = (sup_moins + sup_plus)*1.0     # que ce soit une division entière.
        # calcul de l'entropie de la coupure
        val_entropie_inf = shannon([inf_moins / nb_inf, inf_plus  / nb_inf])
        val_entropie_sup = shannon([sup_moins / nb_sup, sup_plus  / nb_sup])
        val_entropie = (nb_inf / nb_total) * val_entropie_inf \
                       + (nb_sup / nb_total) * val_entropie_sup
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (min_entropie > val_entropie):
            min_entropie = val_entropie
            min_seuil = val_seuil
    return (min_seuil, min_entropie)

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

def divise(LabeledSet, att, seuil):
    Lb1 = ls.LabeledSet(LabeledSet.getInputDimension())
    Lb2 = ls.LabeledSet(LabeledSet.getInputDimension())
    
    for i in range(LabeledSet.size()):
        if LabeledSet.getX(i)[att] <= seuil:
            Lb1.addExample(LabeledSet.getX(i), LabeledSet.getY(i))
        else:
            Lb2.addExample(LabeledSet.getX(i), LabeledSet.getY(i))
        
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
        #Aqui?
        feuille.ajoute_feuille(classe_majoritaire(LSet))
        return feuille
    noeud = ArbreBinaire()
    LSGauche, LSDroite = divise(LSet, att, seuil)   
    try:
        ADGauche = construit_AD(LSGauche, epsilon, level = level + 1, maxLevel = maxLevel)
    except AttributeError:
        print("LSet.size() =",LSet.size(), "entropie(LSet) =",entropie(LSet),"epsilon", epsilon, "gauche:",LSGauche.size())
        print("att =",att,"entro =",entro, "seuil =",seuil)
        LSet.affiche_base()
        print("################################################################################")
        raise AttributeError()
    try:
        ADDroite = construit_AD(LSDroite, epsilon, level = level + 1, maxLevel = maxLevel)
    except AttributeError:
        print("LSet.size() =",LSet.size(), "entropie(LSet) =",entropie(LSet),"epsilon", epsilon,"droite:",LSDroite.size())
        print("att =",att,"entro =",entro, "seuil =",seuil)
        LSet.affiche_base()
        print("################################################################################")
        raise AttributeError()
    noeud.ajoute_fils(ADGauche, ADDroite, att, seuil)
    return noeud

myLSet = ls.LabeledSet(9)
myLSet.addExample([-0.76347689, 1.88311588, -0.65673684, -0.32301959, -0.3599601, 0.6967062, -0.81301642, -0.48597031, 0.14895667], -1)
myLSet.addExample([-0.76347689, 0.35908308, -0.34597325, -0.52634173, -0.45268589, -0.14424948, -0.51679855, 0.06700895, 0.25042737], 1)
myLSet.addExample([-2.06679133, -1.77527287, -0.43476285, -0.38788005, -0.37478621, -0.92450651, 3.78165056, -0.63563805, 0.28207409], -1)
myLSet.addExample([-1.57029059, 1.77894125, -0.47915765, 0.39868955, 0.08648773, 1.18632492, -0.74245513, 0.19177888, 0.15556817], -1)
myLSet.addExample([-2.00472873, -0.94063409, -0.70113164, -0.58822238, -0.51282118, -0.32137756, 1.97113152, -0.77364539, -0.63864965], -1)



