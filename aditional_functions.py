# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

# Funciones adicionales necesarias para el correcto funcionamiento del 
# algoritmo

import math
import numpy as np
import random
from scipy.optimize import minimize
from scipy.optimize import lsq_linear
import matlab.engine

# Herramienta para ejecturar funciones de Matlab
eng=matlab.engine.start_matlab()
eng.addpath(r'C:\Users\Dani Vela\Desktop\TFM\Codes\Matlab\FinalCode', nargout=0)

# Devuelve una matriz donde se almacenan las distancias entre los sensores o 
# los nodos
def calculate_real_distance(Nall, posNx, posNy):
    
    Dall=np.zeros((Nall,Nall))
    
    for ii in range(0,Nall):
        for jj in range(0,Nall):
            Dall[ii,jj]=math.sqrt((posNx[ii]-posNx[jj])**2+(posNy[ii]-posNy[jj])**2)
            
    return Dall

# Genera las muestras para la atenuación debido al fenómeno de 
# "shadowing" la cual sigue una distribución normal de media cero y
# varianza shadowvar. Estas muestras tienen que ser correladas por
# por lo que son generadas con una variable gaussiana multidimensional
# de media cero y covarianza c    
def calculate_sample_shadow(variables, Nall, Dall, kk):
    
    # Media 0
    m=np.zeros((Nall))
    # Calculo de \rho_u
    rho=np.exp(-Dall/variables.Dcorr)
        
    shadowvar=variables.shadowvarSim[kk]
    shadowvar_nu=np.power(10,shadowvar/10)
    shadowstd_nu=math.sqrt(shadowvar_nu)
    shadowstd=10*np.log10(shadowstd_nu)
    
    # En cada iteración se tiene que calcular este valor. 
    #Parámetro: \sigma_v^{2}
    var_v=shadowstd**2
    
    # Ecuación (4) en paper
    C=var_v*rho
    
    # Generación de muestras para atenuación debido al efecto de "shadowing"
    deltaall=np.random.multivariate_normal(m,C)
    
    # Las muestras se distribuyen entre las salidas para las muestras de
    # entrenamiento y las muestras de test
    deltaM=deltaall[0:variables.M*variables.Ttotal]
    deltaU=deltaall[variables.M*variables.Ttotal:Nall]
        
    deltaM=np.reshape(deltaM,(variables.M,variables.Ttotal),order='F')
    
    return deltaM, deltaU, shadowstd, var_v

#Está función se encarga de determinar, en base al estado de la bandera 
#flag_intermitent, los sensores que están activos en el instante de tiempo 
#actual así como su posición
def readjust_node_emition(flags, variables, setup, t, nexperiments):
    global posNx
    global posNy
    global indexon
    if flags.flag_intermitent==1:
        if variables.rateoff==0:
            variables.Moff=0
        else:
            # Número de nodos que están apagados en este instante
            # de tiempo
            variables.Moff=np.ceil(variables.M*variables.rateoff/100)
            variables.Mon=int(variables.M-variables.Moff)
            Prev=np.asarray(random.sample(range(variables.M), int(variables.Mon)))
            Prev.ravel().sort()
            indexon=Prev
    else:
        # Índice de los nodos que están disponibles
        indexon=np.arange(variables.M)
        # Incialmente, todos los nodos están disponibles 
        variables.Mon=variables.M
            
    # Número total de nodos
    variables.N=variables.U+variables.Mon;
            
    setup.posMx=setup.posMxall[indexon,t,nexperiments]
    setup.posMy=setup.posMyall[indexon,t,nexperiments]
    posNx=np.append(setup.posMx, setup.posUx)
    posNy=np.append(setup.posMy, setup.posUy)
        
    return posNx, posNy, indexon

# Cálculo de la distancia de ruido
def calculate_noise_distance(variables, posNxNoise, posNyNoise):
    DNoise=np.zeros((variables.N,variables.N))
    for ii in range(0,variables.N):
        for jj in range(0,variables.N):
            DNoise[ii,jj]=np.sqrt((posNxNoise[ii]-posNxNoise[jj])**2+
                                  (posNyNoise[ii]-posNyNoise[jj])**2)
            
    return DNoise

# Estimación de la posición del transmisor
# NOTA: No incluye el refinamiento de la estimación 
def estimate_pos_tx(ZMon, posMxNoise, posMyNoise, posNxNoise, posNyNoise, t):
    global w_told
    global postx_est
    # Pesos. Ecuación (12) 
    wi=10**(ZMon/10)
    # Algoritmo recursivo para estimar la posición del transmisor
    if t>0:
        # Recálculo de los pesos en cada instante de tiempo
        # Ecuación (13)
        w_t=w_told+np.sum(wi)
        # Uso de los pesos para estimar la posición del transmisor
        # Ecuación (11)
        postx_est=np.append((np.sum(wi*posMxNoise)+w_told*postx_est[0])/w_t,
                            (np.sum(wi*posMyNoise)+w_told*postx_est[1])/w_t)
        w_told=w_t
    else:
        postx_est=np.append(np.sum(wi*posMxNoise)/np.sum(wi),
                            np.sum(wi*posMyNoise)/np.sum(wi))
        w_told=np.sum(wi)
                
    # Se incluye el error en la localización del transmisor debido 
    # al ruido de la distancia de los senores y los "grid points"
    dtxNoise=np.sqrt((postx_est[0]-posNxNoise)**2+
                     (postx_est[1]-posNyNoise)**2)
        
    return dtxNoise, postx_est

# Estimación de la potencia transmitida (P) y del exponente de 
# pérdidas por camino (\alpha)
# NOTA: Incluye el refinamiento correspondiente a la estimación de la posición
# del tranmisor y la consiguiente re-estimación de la potencia transmitida y 
# del exponente de pérdidas por camino
def alpha_beta_estimation(flags, variables, dtxNoise, CNoise, ZMon, ZUon,
                         posMxNoise, posMyNoise, posNyNoise, posNxNoise, 
                         postx_est):
    
    # Variable \hat{q} de la Ecuación (25)
    q2use=10*np.log10(dtxNoise[0:variables.Mon])
    Sigmazalpha=CNoise[0:variables.Mon,0:variables.Mon]+(variables.varnoise+variables.noise_varnoise)*np.eye(variables.Mon)+np.diag((variables.sigma_u+variables.noise_sigmau)**2/dtxNoise[0:variables.Mon]**2)
                
    # Valor mínimo permitido para alpha
    minalpha=2
    # Variable \sqrt{\hat{D}}^{-1} en Ecuación (25)
    errloc=1./dtxNoise[0:variables.Mon]
    
    # Matriz correspondiente al primer sumando de la Ecuación (25)
    Prev=np.append(np.ones(variables.Mon)/errloc, -q2use/errloc)
    Prev2=np.reshape(Prev,(variables.Mon,2),order='F')
    # Minimización correspondiente a la Ecuación (25)
    params=lsq_linear(np.reshape(Prev,(variables.Mon,2),order='F'), 
                      ZMon[0:variables.Mon]/errloc)
    
    # Misma operación pero con función lsqlin de MATLAB
    #params=eng.lsqlin(matlab.double(np.array(Prev2).tolist()),
    #                  matlab.double(np.array(ZMon[0:Mon]/errloc).tolist()),
    #                  matlab.double(np.array([[0,0],[0,-1]]).tolist()),
    #                  matlab.double(np.array([0,-minalpha]).tolist()))
                                  
    meanP=params.x[0]
    #meanP=np.array(params)[0][0]
    meanalpha=params.x[1]
    #meanalpha=np.array(params)[1][0]
     
    # Variable \hat{\mu}_{z} en PAG 4    
    meanz=meanP-q2use*meanalpha
    # Variable \Sigma_{z}
    Sigmaz=(ZMon[0:variables.Mon]-meanz)[:,np.newaxis]@(ZMon[0:variables.Mon]-meanz)[np.newaxis,:]
            
    # Variable A en PAG 4
    AAon=Sigmaz-Sigmazalpha
    # Variable \hat{Q} en PAG 4
    BBon=q2use[:,np.newaxis]@q2use[np.newaxis,:]
            
    # Se obtiene las diagonales de las dos matrices anteriores
    aaon=np.diag(AAon).flatten('F')
    bbon=np.diag(BBon).flatten('F')
            
    # Matriz correspondiente al primer sumando de la ecuación (26)
    Prev=np.append(np.ones(np.size(aaon)), bbon)
    Prev2=np.reshape(Prev,(np.size(aaon),2), order='F')
    # Minimización correspondiente a la Ecuación (26)
    #params=lsq_linear(np.reshape(Prev,(np.size(aaon),2), order='F'),
    #                  aaon, bounds=(0,np.inf))
    
    # Misma operación pero con función lsqlin de MATLAB
    params=eng.lsqlin(matlab.double(np.array(Prev2).tolist()),
                      matlab.double(np.array(aaon).tolist()),
                      matlab.double(np.array([[-1,0],[0,-1]]).tolist()),
                      matlab.double(np.array([0,0]).tolist()))
            
    #varP=params.x[0]
    varP=np.array(params)[0][0]
    #varalpha=params.x[1]
    varalpha=np.array(params)[1][0]
            
    RSSourparameters=meanP-10*meanalpha*np.log10(dtxNoise[variables.Mon:variables.N])
    mseourparameters=np.sum((RSSourparameters-ZUon)**2)/variables.U
           
    # Refinamiento de la posición del transmisor
    if flags.flag_tx_est==1:
        rmean=np.sum(ZMon)/variables.Mon
        betai=lambda ss, tt: np.log10(np.sqrt((ss-posMxNoise)**2+
                                              (tt-posMyNoise)**2))
        betamean=lambda ss, tt: np.sum(betai(ss,tt))/variables.Mon
        fun=lambda sstt: np.sum((betai(sstt[0],sstt[1])-betamean(sstt[0],sstt[1]))*
                                (ZMon-rmean))/np.sqrt(
                                    np.sum((betai(sstt[0],sstt[1])-
                                            betamean(sstt[0], sstt[1]))**2)*
                                            np.sum((ZMon-rmean)**2))
        x0=postx_est
        # Minimización correspondiente a la Ecuación (27)
        postx_est_prev=minimize(fun, x0)
        postx_est=postx_est_prev.x        
        dtxNoise=np.sqrt((postx_est[0]-posNxNoise)**2+
                         (postx_est[1]-posNyNoise)**2)
                
        # Restimación de la media de alpha y P
        q2use=10*np.log10(dtxNoise[0:variables.Mon])
        errloc=1./dtxNoise[0:variables.Mon]
        Prev=np.append(np.ones(variables.Mon)/errloc, -q2use/errloc)
        params=lsq_linear(np.reshape(Prev,(variables.Mon,2),order='F'), 
                          ZMon[0:variables.Mon]/errloc)
        meanP=params.x[0]
        meanalpha=params.x[1]
        
    return meanP, meanalpha, varalpha, varP 