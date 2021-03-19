# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

# Funciones correspondientes a la ejecución de los algoritmos presentes en 
# el trabajo

import numpy as np
import matlab.engine
import GPy


# Herramienta para ejecturar funciones de Matlab
eng=matlab.engine.start_matlab()
eng.addpath(r'C:\Users\Dani Vela\Desktop\TFM\Codes\Matlab\FinalCode', nargout=0)

# Uso de algoritmo de Gholami para estimación de parámetros
def algoritmo_Gholami(variables, setup, posMxNoise, posMyNoise, 
                      posNyNoise, posNxNoise, ZMon, kk, t, ZUon):
    
    Prev=np.append(posMxNoise, posMyNoise)
    Prev2=np.reshape(Prev,(np.size(posMxNoise),2),order='F')
    meanPGholami,meanalphaGholami,postx_estGholami=eng.Gholami(       
    matlab.double(np.array(ZMon[:,np.newaxis]).tolist()),
    matlab.double(np.array(Prev2).tolist()),nargout=3)
                
    dtxNoiseGholami=np.sqrt((postx_estGholami[0][0]-posNxNoise)**2 + (postx_estGholami[1][0]-posNyNoise)**2)
                
    RSSGholami=meanPGholami-10*meanalphaGholami*np.log10(dtxNoiseGholami[variables.Mon:variables.N])
    setup.mseGholami[kk,t]=setup.mseGholami[kk,t]+np.sum((RSSGholami-ZUon)**2)

# Uso de algoritmo de Watcharapan para estimación de parámetros
def algoritmo_watcharapan(variables, setup, posNyNoise, posNxNoise,
                          ZUon, ZMon, kk, t):
    
    meanPWatcharapan,meanalphaWatcharapan,postx_estWatcharapan=eng.Watcharapan(       
    matlab.double(np.array(ZMon).tolist()),
    matlab.double(np.array(posNxNoise[0:variables.M]).tolist()),
    matlab.double(np.array(posNyNoise[0:variables.M]).tolist()),nargout=3)                                
            
    dtxNoiseWatcharapan=np.sqrt((postx_estWatcharapan[0][0]-posNxNoise)**2 + (postx_estWatcharapan[0][1]-posNyNoise)**2)
            
    RSSWatcharapan=meanPWatcharapan-10*meanalphaWatcharapan*np.log10(dtxNoiseWatcharapan[variables.Mon:variables.N])
    setup.mseWatcharapan[kk,t]=setup.mseWatcharapan[kk,t]+np.sum((RSSWatcharapan-ZUon)**2)

# Uso del algoritmo OKD para estimación de campo recibido
def algoritmo_OKD(variables, setup, meanP, meanalpha, dtxNoise, ZMon, 
                  DNoise, ZUon, kk, t):
    
    # Se calcula L en todas las localizaciones
    Lest=meanP-10*meanalpha*np.log10(dtxNoise)          
                
    # Se predice el valor en las localizaciones no disponibles
    deltaEst=ZMon[0:variables.Mon]-Lest[0:variables.Mon]
                
    Cok=5
    a=100
    gamma=lambda h: Cok*(1-np.exp(-h/a))
    GAMMA=gamma(DNoise[0:variables.Mon,0:variables.Mon])
                
    for k in range(0,variables.U):
        GAMMAu=gamma(DNoise[0:variables.Mon,variables.Mon+k])
        weights=np.linalg.pinv(np.block([[GAMMA,np.ones((variables.Mon,1))], [np.ones((1, variables.Mon)), 0]]))@np.append(GAMMAu,1)
        deltau=np.sum(weights[0:variables.Mon]*deltaEst[0:variables.Mon])
        setup.RSSok[k]=deltau+Lest[variables.Mon+k]
                    
    setup.mseok[kk,t]=setup.mseok[kk,t]+np.sum((setup.RSSok.flatten('F')-ZUon)**2)
  
# Uso del algoritmo GPR para estimar el campo recibido en el caso en el que la 
# información proporcionada por los sensores varía con el tiempo
def algoritmo_GPRecursive(flags, variables, setup, posNxNoise, posNyNoise, ZMon, 
                          dtxNoise, meanalpha, meanP, varalpha, varP, shadowstd, 
                          ZUon, kk, t):
    
    global mprior
    global Cprior
    Prev=np.append(posNxNoise[0:variables.Mon], posNyNoise[0:variables.Mon])
    # Entradas correspondientes a las muestras de entrenamiento
    Xtrain=np.reshape(Prev, (variables.Mon, 2), order='F') 
    N_training=np.size(Xtrain,0)
    # Salidas correspondientes a las muestras de entrenamiento
    ytrain=ZMon
    Prev=np.append(posNxNoise[variables.Mon:variables.N], posNyNoise[variables.Mon:variables.N])
    # Entradas correspondientes a las muestras de entrenamiento
    Xtest=np.reshape(Prev, (variables.N-variables.Mon,2), order='F')
                
    q=10*np.log10(dtxNoise[0:variables.Mon])
    qd=10*np.log10(dtxNoise[variables.Mon:variables.N])
    # Variable que trata de emular los kernel Bias y LOG 
    qq=varalpha*np.append(q,qd)[:,np.newaxis]@np.append(q,qd)[np.newaxis,:]+varP*np.ones([variables.N,variables.N])
    Sigman=(variables.varnoise+variables.noise_varnoise)*np.eye(variables.Mon)+ np.diag((variables.sigma_u+variables.noise_sigmau)**2/dtxNoise[0:variables.Mon]**2)
    # Media de GP
    mtrain=meanP-10*meanalpha*np.log10(dtxNoise[0:variables.Mon])
    mtest=meanP-meanalpha*qd
            
    if flags.optimiseHyperparameters==True:
        # Creación del Kernel
        #biasKern=GPy.kern.Bias(Xtrain.shape[1])
        bias=varP*np.ones([variables.N,variables.N])
        NSEKern=GPy.kern.NSE(Xtrain.shape[1])
        LOGKern=GPy.kern.LOG(Xtrain.shape[1])
        
        # Suma de los tres kernels. Ecuación (42)
        kern=NSEKern+LOGKern#+biasKern
        
        # Ruido
        beta = np.diag(1/((variables.sigma_u+variables.noise_sigmau)**2/dtxNoise**2+(variables.varnoise+variables.noise_varnoise)*np.ones(variables.N)))

        Prev=(ytrain-mtrain)[:,np.newaxis]@np.ones([1,2])
        
        # Selección de modelo Gaussian Processes for Regression
        mR=GPy.models.GPRegression(Xtrain, Prev, kern)
        # Optimización de los hiperparámetros
        mR.optimize()
        
        # Obtención del kernel una vez optimizado los hiperparámetros            
        kern=mR.kern
        
        # K_{\hat{X}} en Ecuación (37)
        Ktrain=kern.K(Xtrain,Xtrain)+bias[0:N_training,0:N_training]+beta[0:N_training,0:N_training]
        # K_{X_{g}, \hat{X}} en Ecuación (37)
        Kx=kern.K(Xtest,Xtrain)+bias[N_training:,0:N_training]+beta[N_training:,0:N_training]
        # K_{X_{g}} en Ecuación (37)
        Ktest=kern.K(Xtest,Xtest)+bias[N_training:,N_training:]+beta[N_training:,N_training:]
        # Inversa de la Ecuación (41)
        C_inv=np.linalg.inv(Ktrain+Sigman)
     
    else:   
        # Kernel RBF para el caso en el que no se optimizan los hiperparámetros
        kern=GPy.kern.RBF(input_dim=2,variance=shadowstd**2, 
                          lengthscale=variables.Dcorr, inv_l=True)
                    
        # En este caso se añade la variable qq para tratar de simular lo 
        # máximo posible el kernel de la Ecuación (42)
        Ktrain=kern.K(Xtrain,Xtrain)+qq[0:N_training,0:N_training]
        Kx=kern.K(Xtest,Xtrain)+qq[N_training:,0:N_training]
        Ktest=kern.K(Xtest,Xtest)+qq[N_training:,N_training:]
        C_inv=np.linalg.inv(Ktrain+Sigman)
     
    # \mu_{post}. Ecuación (57)
    mpos=Kx@C_inv@(ytrain-mtrain)
    # \Sigma_{post}. Ecuación (58)
    Cpos=Kx@C_inv@Kx.T
                
    # "prior" in los nodos
    if t==0 or (flags.flag_moving==0 and flags.flag_intermitent==0):
        mprior=mpos
        Cprior=Cpos
                       
    # La media de la predicción 
    mGP=mtest+variables.forgetting_factor*mprior+(1-variables.forgetting_factor)*mpos
    #  La covarianza de la predicción
    vGP=Ktest-variables.forgetting_factor*Cprior-(1-variables.forgetting_factor)*Cpos
                        
    setup.RSSgprecursive=mGP
    
    #\mu_{prior}. Ecuación (59)
    mprior=variables.forgetting_factor*mprior+(1-variables.forgetting_factor)*mpos
    #\Sigma_{prior}. Ecuación (60)
    Cprior=variables.forgetting_factor*Cprior+(1-variables.forgetting_factor)*Cpos
                
    #Error cuadrático medio entre señal estimada y valor teórico
    setup.mseGPrecursive[kk,t]=setup.mseGPrecursive[kk,t]+np.sum((setup.RSSgprecursive-ZUon)**2)

# Uso del algoritmo GPR para estimar el campo recibido en el caso en el que la 
# información proporcionada por los sensores es constante en el tiempo
def algoritmo_GPStatic(flags, variables, setup, posNxNoise, posNyNoise, ZMon, 
                          dtxNoise, meanalpha, meanP, varalpha, varP, shadowstd, 
                          ZUon, kk, t):
    
    Prev=np.append(posNxNoise[0:variables.Mon], posNyNoise[0:variables.Mon])
    # Entradas correspondientes a las muestras de entrenamiento
    Xtrain=np.reshape(Prev, (variables.Mon, 2), order='F') 
    N_training=np.size(Xtrain,0)
    # Salidas correspondientes a las muestras de entrenamiento
    ytrain=ZMon
    Prev=np.append(posNxNoise[variables.Mon:variables.N], posNyNoise[variables.Mon:variables.N])
    # Entradas correspondientes a las muestras de test
    Xtest=np.reshape(Prev, (variables.N-variables.Mon,2), order='F')
                
    q=10*np.log10(dtxNoise[0:variables.Mon])
    qd=10*np.log10(dtxNoise[variables.Mon:variables.N])
    # Variable que trata de emular los kernel Bias y LOG 
    qq=varalpha*np.append(q,qd)[:,np.newaxis]@np.append(q,qd)[np.newaxis,:]+varP*np.ones([variables.N,variables.N])
    Sigman=(variables.varnoise+variables.noise_varnoise)*np.eye(variables.Mon)+ np.diag((variables.sigma_u+variables.noise_sigmau)**2/dtxNoise[0:variables.Mon]**2)
    # Media de GP
    mtrain=meanP-10*meanalpha*np.log10(dtxNoise[0:variables.Mon])
    mtest=meanP-meanalpha*qd
            
    if flags.optimiseHyperparameters==True:
        # Creación del Kernel
        #biasKern=GPy.kern.Bias(Xtrain.shape[1])
        bias=varP*np.ones([variables.N,variables.N])
        NSEKern=GPy.kern.NSE(Xtrain.shape[1])
        LOGKern=GPy.kern.LOG(Xtrain.shape[1])
    
        # Suma de los tres kernels. Ecuación (42)
        kern=NSEKern+LOGKern#+biasKern
        #beta = np.diag(1/((sigma_u+noise_sigmau)**2/dtxNoise[0:Mon]**2+(varnoise+noise_varnoise)*np.ones(Mon)))
        beta = np.diag(1/((variables.sigma_u+variables.noise_sigmau)**2/dtxNoise**2+(variables.varnoise+variables.noise_varnoise)*np.ones(variables.N)))
                   
        Prev=(ytrain-mtrain)[:,np.newaxis]@np.ones([1,2])

        if flags.GPRegression==1:
            # Selección de modelo Gaussian Processes for Regression
            mR=GPy.models.GPRegression(Xtrain, Prev, kern)
            # Optimización de los hiperparámetros
            mR.optimize()
             
            # Obtención del kernel una vez optimizado los hiperparámetros            
            kern=mR.kern
            
            # K_{\hat{X}} en Ecuación (37)    
            Ktrain=kern.K(Xtrain,Xtrain)+bias[0:N_training,0:N_training]+beta[0:N_training,0:N_training]
            # K_{X_{g}, \hat{X}} en Ecuación (37)
            Kx=kern.K(Xtest,Xtrain)+bias[N_training:,0:N_training]+bias[N_training:,0:N_training]
            # K_{X_{g}} en Ecuación (37)
            Ktest=kern.K(Xtest,Xtest)+bias[N_training:,N_training:]+bias[N_training:,N_training:]
            # Inversa de la Ecuación (41)
            C_inv=np.linalg.inv(Ktrain+Sigman)
                        
            # La media de la predicción 
            mGP=mtest+Kx@C_inv@(ytrain-mtrain)
            #  La covarianza de la predicción
            vGP=Ktest-Kx@C_inv@Kx.T
                   
            setup.RSSgpstatic=mGP
                       
            #Error cuadrático medio entre señal estimada y valor teórico
            setup.mseGPRstatic[kk,t]=setup.mseGPRstatic[kk,t]+np.sum((setup.RSSgpstatic-ZUon)**2)
                    
        if flags.GPRegression_inducing==1:
            num_inputs=150
            # Selección de modelo Gaussian Processes for Regression con inducing inputs
            mC=GPy.models.SparseGPRegression(Xtrain, Prev, kern, num_inducing=num_inputs)
            # Optimización de los hiperparámetros
            mC.optimize
                
            # Obtención del kernel una vez optimizado los hiperparámetros            
            kern=mC.kern
            
            # K_{\hat{X}} en Ecuación (37)
            Ktrain=kern.K(Xtrain,Xtrain)
            # K_{X_{g}, \hat{X}} en Ecuación (37)
            Kx=kern.K(Xtest,Xtrain)
            # K_{X_{g}} en Ecuación (37)
            Ktest=kern.K(Xtest,Xtest)
            # Inversa de la Ecuación (41)
            C_inv=np.linalg.inv(Ktrain+Sigman)
                        
            # La media de la predicción 
            mGP=mtest+Kx@C_inv@(ytrain-mtrain)
            #  La covarianza de la predicción
            vGP=Ktest-Kx@C_inv@Kx.T
                    
            setup.RSSgpstatic=mGP
                        
            #Error cuadrático medio entre señal estimada y valor teórico
            setup.mseGPCstatic[kk,t]=setup.mseGPCstatic[kk,t]+np.sum((setup.RSSgpstatic-ZUon)**2)

    else:
        # Kernel RBF para el caso en el que no se optimizan los hiperparámetros
        kern=GPy.kern.RBF(input_dim=2,variance=shadowstd**2, 
                          lengthscale=variables.Dcorr, inv_l=True)
        
        # En este caso se añade la variable qq para tratar de simular lo 
        # máximo posible el kernel de la Ecuación (42)                                
        Ktrain=kern.K(Xtrain,Xtrain)+qq[0:N_training,0:N_training]
        Kx=kern.K(Xtest,Xtrain)+qq[N_training:,0:N_training]
        Ktest=kern.K(Xtest,Xtest)+qq[N_training:,N_training:]
        C_inv=np.linalg.inv(Ktrain+Sigman)
                    
        # La media de la predicción 
        mGP=mtest+Kx@C_inv@(ytrain-mtrain)
        #  La covarianza de la predicción
        vGP=Ktest-Kx@C_inv@Kx.T
                    
        setup.RSSgpstatic=mGP
                
        #Error cuadrático medio entre señal estimada y valor teórico
        setup.mseGPstatic[kk,t]=setup.mseGPstatic[kk,t]+np.sum((setup.RSSgpstatic-ZUon)**2)
            