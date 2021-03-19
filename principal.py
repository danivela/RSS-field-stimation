# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

# Código Python correspondiente al paper:
# "Recursive Estimation of Dynamic RSS Fields Based on 
# Crowdsourcing and Gaussian Processes"

# Script principal encargado de llamar a las distintas funciones que forman
# este proyecto

# Librerías importadas
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import lsq_linear
import GPy
import matlab.engine
from flags import Flags
from variable import Variables
from setup import Setup
import aditional_functions as functions
import graficas as gr
import algoritmos as alg
#import gpflow

# Herramienta para ejecturar funciones de Matlab
eng=matlab.engine.start_matlab()
eng.addpath(r'C:\Users\Dani Vela\Desktop\TFM\Codes\Matlab\FinalCode', nargout=0)


#%%

# Sentencia necesario para medir el tiempo de ejecución de las simulaciones
expend=time.time()

'''
Los distintos flags son almacenados en la siguiente clase:
    1. Escenario usado. Hay tres posibles escenarios:
        1.1. Todos los sensores están estáticos (1)
        1.2. Todos los sensores se están moviendo (2)
        1.3. Todos los sensores están estáticos pero no están siempre transmitiendo (3)
    2. Estimación de la posición del transmisor (1 ó 0)
    3. Uso de algorimto GPR para escenario estático (1 ó 0)
    4. Uso de algoritmo GPR para escenario recursivo (1 ó 0)
    5. Uso del algoritmo OKD. Mirar referencia del paper (1 ó 0)
    6. Uso de algoritmo "Gholami". Mirar referencia del paper (1 ó 0)
    7. Uso de algoritmo "Watcharapan". Mirar referencia del paper (1 ó 0)
    8. Representación de las figuras (1 ó 0)
    9. Representación de los hiperparámetros estimados del kernel (1 ó 0)
    10. Posición fija de los nodos (1 ó 0)
    11. Los hiperparámetros se obtienen del Kernel (True) o de los datos (False)
    12. El modelo usado es GP para regresión (1 ó 0)
    13. El modelo usado es GP para regresión con inducing inputs (1 ó 0)
'''
flags=Flags(1, 0, 1, 0, 0, 0, 0, 1, 0, 1, True, 1, 0)
flags.flag_mov_inter()

# Establecemos una semilla para controlar la aleatoriedad de las muestras
np.random.seed(1) 

''' 
Las variables que han sido obtenidas en distintos estudios son almacenados en 
la siguien clase, además de las variables necesarias para la ejecución del 
programa:
    1. Varianza del efecto "shadowing" en el exterior en dB
    2. Número total de experimentos
    3. Instantes de tiempo simulado
    4. Potencia transmitida (dBm)
    5. Número de nodos no disponibles
    6. Pérdidas por multitrayecto
    7. rho_u en el paper (mdB)
    8. Distancia de decorrelación
    9. Varianza de ruido
    10. "Forgetting Factor". Lambda = 0, olvida todo lo anterior. Lambda = 1 
    todas las muestras anteriores aportan lo mismo 
    11. Porcentaje de los nodos totales que estarán off en cada instante de tiempo
    12. Varianza entre instantes de tiempo consecutivos en el que se mueve los nodos
    13. noise_sigmau
    14. noise_varnoise
'''
variables=Variables([0.5, 2, 4, 6, 8, 10], 10, 10, -10, 10, 3.5, 200, 50, 
                   7, 0.5, 20, 1, 0, 0)

''' 
Clase para preparar el setup necesario para este trabajo:
    1. Grid donde se encuentran los nodos (500mx500m)
    *La densidad de nodos de observación es de 4 por cada 100mx100m
    2. Densidad de nodos dentro del grid
    3. Distancia entre nodos
'''
setup=Setup([500, 500], 0.2, 100)

# Posición de los nodos
setup.node_position(flags, variables)

# Posición del transmisor
# Situación del transmisor en el centro del grid
postx=[0, 0]
setup.tx_position(postx, variables)

# Distribución de los nodos disponibles entre el total de nodos del Grid
setup.node_distribution(flags, variables)

# A continuación se inicializa los vectores donde se almacenará el error
# cuadrático medio para cada algoritmo
setup.initialize_vectors(flags, variables)
   
# Bucle principal 
for nexperiments in range(0,variables.Nexp): # Nexp
    print ('Nexp=' + str(nexperiments) + '/' + str(variables.Nexp-1))
    
    # Para cada experimento, genera los datos. Vector con la posición de los
    # sensores y de todos los grid
    posNx=np.append(setup.posMxall[:,:,nexperiments].flatten('F'), setup.posUx)
    posNy=np.append(setup.posMyall[:,:,nexperiments].flatten('F'), setup.posUy)

    # Calculo de las distancias real
    Nall=len(posNx)
    Dall=functions.calculate_real_distance(Nall, posNx, posNy)
    
    #%%
    for kk in range(0,len(variables.shadowvarSim)): #len(shadowvarSim)
        print ('ShadowvarSim=' + str(kk) + '/' + str(len(variables.shadowvarSim)-1))
        # Genera las muestras para la atenuación debido al fenómeno de 
        # "shadowing" la cual sigue una distribución normal de media cero y
        # varianza shadowvar. Estas muestras tienen que ser correladas por
        # por lo que son generadas con una variable gaussiana multidimensional
        # de media cero y covarianza c
        deltaM, deltaU, shadowstd, var_v = functions.calculate_sample_shadow(
            variables, Nall, Dall, kk)
        
    #%% 
        for t in range(0,variables.Ttotal): #Ttotal
            print ('Instante=' + str(t) + '/' + str(variables.Ttotal-1))
            
            # En función del instante de tiempo, algunos nodos estarán emitiendo
            # o no. Esto se refleja en los nuevos valores de posNx y posNy
            posNx, posNy, indexon = functions.readjust_node_emition(flags, variables, setup, t, nexperiments)
            
            dtx=np.sqrt((postx[0]-posNx)**2+(postx[1]-posNy)**2)
            dtxNoise=np.append(np.absolute(dtx[0:variables.Mon]+variables.sigma_d*np.random.randn(variables.Mon)),dtx[variables.Mon:variables.N])
            # Generación a partir de la función randn de MATLAB
            #dtxNoise=np.append(np.absolute(dtx[0:Mon]+sigma_d*np.array(eng.randn(1,Mon))[0]),dtx[Mon:N])
            
            # Error en la posición
            errd=dtxNoise[0:variables.Mon]/dtx[0:variables.Mon]
            posMxNoise=posNx[0:variables.Mon]*errd
            posMyNoise=posNy[0:variables.Mon]*errd
            
            posNxNoise=np.append(posMxNoise, setup.posUx)
            posNyNoise=np.append(posMyNoise, setup.posUy)
            
            # Calculo de la "distancia de ruido" entre posiciones
            DNoise=functions.calculate_noise_distance(variables, posNxNoise, posNyNoise)
       
    #%%
            CNoise=var_v*np.exp(-DNoise/variables.Dcorr)
            
            # Datos
            
            # Path-loss exponent
            alpha=np.ones(variables.N)*variables.alphai
            
            # Shadowing
            delta=np.append(deltaM[indexon,t],deltaU)
            
            # Ruido
            noise=np.append(np.sqrt(variables.varnoise)*np.random.randn(variables.Mon),np.zeros(variables.U))
            
            # RSSI teórico
            Z=variables.P-np.diag(10*alpha*np.log10(dtx))@np.ones(variables.N)+delta+noise
            ZMon=Z[0:variables.Mon]
            ZUon=Z[variables.Mon:variables.N]
            
            if flags.flag_tx_est == 1:
                # Estimación de la posición del transmisor
                dtxNoise, postx_est = functions.estimate_pos_tx(ZMon, posMxNoise, 
                                    posMyNoise, posNxNoise, posNyNoise, t)                                
            else:
                postx_est=postx
                
            # Representación únicamente en la primera iteración    
            if flags.flagfigs==1 and t==0 and kk==0 and nexperiments==0:
                # Figura 2 en el paper
                gr.representacion_scenario(setup, postx, ZMon)
            
    #%%
            # Estimación de alfa y de P
            meanP, meanalpha, varalpha, varP = functions.alpha_beta_estimation(
            flags, variables, dtxNoise, CNoise, ZMon, ZUon, posMxNoise, posMyNoise, 
            posNyNoise, posNxNoise, postx_est)

#%%         
            if flags.flagfigs==1 and t==0 and kk==0 and nexperiments==0:
                # Figura 4 en el paper
                gr.representacion_variacion_potencia(variables, ZMon, dtxNoise, 
                                                     meanP, meanalpha)
   
            # A continuación iría uno por cada algoritmo que se quiera hacer
            #%%    
            # Gholami
            if flags.flagGholami==1:
                # Estimación a partir de método de Gholami [Mirar referencias en la memoria]
                alg.algoritmo_Gholami(variables, setup, posMxNoise, posMyNoise, posNyNoise, posNxNoise, ZMon, kk, t, ZUon)
            #%%
            # Watcharapan
            if flags.flagWatcharapan==1:
                # Estimación a partir de método de Watcharapan [Mirar referencias en la memoria]
                alg.algoritmo_watcharapan(variables, setup, posNyNoise, posNxNoise, ZUon, ZMon, kk, t)
            # %%            
            # Se estima la señal recibida en los nodos disponibles                    
            #OKD
            if flags.flagOKD==1:
                # Estimación a partir de método de OKD [Mirar referencias en la memoria]
                alg.algoritmo_OKD(variables, setup, meanP, meanalpha, dtxNoise, ZMon, DNoise, ZUon, kk, t)
            #%%  
            # GPrecursive
            if flags.flagGPrecursive==1:
                # Estimación a partir del método de GP para campos variantes en el tiempo
                alg.algoritmo_GPRecursive(flags, variables, setup, posNxNoise, 
                posNyNoise, ZMon, dtxNoise, meanalpha, meanP, varalpha, varP, 
                shadowstd, ZUon, kk, t)
            #%%
            # GPstatic
            if flags.flagGPstatic==1:
                # Estimación a partir del método de GP para campos estáticos en el tiempo
                alg.algoritmo_GPStatic(flags, variables, setup, posNxNoise, 
                posNyNoise, ZMon, dtxNoise, meanalpha, meanP, varalpha, varP, 
                shadowstd, ZUon, kk, t)
#%%
                
# Representación de los resultados de error obtenidos a partir de las simulaciones
my_tplot=1
gr.representacion_final_algoritmos(flags, variables, my_tplot, setup)

# Tiempo total de ejecución  
elapsed=time.time()-expend
print("El tiempo es: " + str(elapsed))

