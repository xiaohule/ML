import numpy as np

def play(nb_tours,changerPorte):
    
    matrice_resultats=np.zeros((1,nb_tours))
    i=0
    
    while i<nb_tours:

        porte_voiture=np.random.randint(0,3) #la porte qui cache la voiture est assignée aléatoirement avant l'arrivée du candidat
        portes=[0,1,2]

        premier_choix=np.random.randint(0,3)
        portes.remove(premier_choix) #enlever le premier choix du candidat de la liste des portes

        if premier_choix==porte_voiture:
            portes.remove(portes[np.random.randint(0,2)]) #l'animateur enlève une chèvre au hasard
        else:
            portes=porte_voiture #l'animateur enlève la seule chèvre parmi les deux portes restantes

        if changerPorte:
            deuxieme_choix=portes
        else:
            deuxieme_choix=premier_choix
        
        matrice_resultats[0,i]=deuxieme_choix==porte_voiture
        i+=1

    return matrice_resultats