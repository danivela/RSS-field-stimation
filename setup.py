# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

# Clase correspondiente a la creación e inicialización del setup de este 
# trabajo. Se genera el escenario, calculando las posiciones de los nodos y 
# de los sensores, además de la del transmisor.

import math
import numpy as np

class Setup:
    def __init__(self, gridP, densityM, step):
        self.gridP=gridP
        self.densityM=densityM
        self.step=step
        self.TotNodes=math.floor(self.step/3)
        self.diffposUx=np.linspace(start=-self.gridP[0]/2, stop=self.gridP[0]/2, 
                                   num=self.TotNodes)
        self.diffposUy=np.linspace(start=-self.gridP[1]/2, stop=self.gridP[1]/2, 
                                   num=self.TotNodes)

    def node_position(self, flags, variables):
        if flags.flagGridFix==1:
            self.posUx, self.posUy=np.meshgrid(self.diffposUx, self.diffposUy)
            self.posUx=self.posUx.flatten('F')
            self.posUy=self.posUy.flatten('F')
            variables.U=len(self.posUy)
        else:
            # Disposición de los nodos no disponible de manera aleatoria
            self.posUx=self.gridP[0]*np.random.rand(variables.U)-self.gridP[0]/2
            self.posUy=self.gridP[1]*np.random.rand(variables.U)-self.gridP[1]/2
            
    def tx_position(self, postx, variables):
        indextx1=np.where(self.posUx==postx[0])
        indextx2=np.where(self.posUy==postx[1])
        indexrep=np.intersect1d(indextx1, indextx2)
        # Quita la posición cero que intersecta en ambos arrays, ya que corresponde
        # con la posición del transmisor y ahí no puede existir ningún nodo
        if indexrep.size != 0:
            self.posUx=np.append(self.posUx[0:indexrep[0]],self.posUx[indexrep[0]+1:variables.U])
            self.posUy=np.append(self.posUy[0:indexrep[0]],self.posUy[indexrep[0]+1:variables.U])
            variables.U=variables.U-1
            
    def node_distribution(self, flags, variables):
        # Nodos disponibles
        variables.M=math.ceil(variables.U*self.densityM)
        # Número total de nodos
        variables.N=variables.M+variables.U
        
        self.posMxall=np.zeros((variables.M,variables.Ttotal,variables.Nexp))
        self.posMyall=np.zeros((variables.M,variables.Ttotal,variables.Nexp))

        for k in range(0,variables.Nexp):
            for t in range(0,variables.Ttotal):
                if t == 0:
                    self.posMx=self.gridP[0]*np.random.rand(variables.M)-self.gridP[0]/2
                    self.posMy=self.gridP[1]*np.random.rand(variables.M)-self.gridP[1]/2
                    self.posMxall[:,t,k]=self.posMx
                    self.posMyall[:,t,k]=self.posMy
                else:
                    self.posMxall[:,t,k]=self.posMxall[:,t-1,k]+flags.flag_moving*np.random.randn(variables.M)*math.sqrt(variables.var_movement)
                    self.posMyall[:,t,k]=self.posMyall[:,t-1,k]+flags.flag_moving*np.random.randn(variables.M)*math.sqrt(variables.var_movement)
                    
    def initialize_vectors(self, flags, variables):
        if flags.flagGholami == 1:
            self.mseGholami = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
    
        if flags.flagWatcharapan == 1:
            self.mseWatcharapan = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
 
        if flags.flagOKD == 1:
            self.mseok = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
            self.RSSok = np.zeros((variables.U,1))  
            
        if flags.flagGPstatic == 1:
            if flags.GPRegression == 1:
                self.mseGPRstatic = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
                self.RSSgprstatic = np.zeros((variables.U,1))
            if flags.GPRegression_inducing == 1:
                self.mseGPCstatic = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
                self.RSSgpcstatic = np.zeros((variables.U,1))
            
        if flags.flagGPrecursive == 1:
            self.mseGPrecursive = np.zeros((len(variables.shadowvarSim),variables.Ttotal))
            self.RSSgprecursive = np.zeros((variables.U,1))