# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

# Funciones correspondientes a la representación de gráficas del trabajo

import matplotlib.pyplot as plt
import numpy as np

# Figura que muestra la disposición de nodos y sensores en el escenario
def representacion_scenario(setup, postx, ZMon):
    plt.plot(setup.posUx, setup.posUy, "s", markersize=3.5, markerfacecolor=
             [0.6, 0.6,0.6], markeredgecolor=[0.6, 0.6, 0.6], zorder=1, label='Nodos')
    plt.plot(postx[0], postx[1], 'g^', markersize=4.5, markerfacecolor=[0, 1, 0], 
             zorder=3, label='Transmisor')
    sens=plt.scatter(setup.posMx, setup.posMy, 30, ZMon, zorder=2, label='Sensores')
    plt.clim(-104,-46)
    h=plt.colorbar()
    plt.xlabel('Eje x (m)')
    plt.ylabel('Eje y (m)')
    h.set_label('RSS [dBm]')
    plt.axis([-setup.gridP[0]/2-setup.step/10, setup.gridP[0]/2+setup.step/10,
              -setup.gridP[1]/2-setup.step/10, setup.gridP[1]/2+setup.step/10])
    plt.legend(loc=1, fontsize="small")
    plt.savefig("escenario_gpr.eps")
    #plt.show()

# Figura que muestra la variación de la señal recibida con la distancia
def representacion_variacion_potencia(variables, ZMon, dtxNoise, meanP, meanalpha):
    fig,ax=plt.subplots(1)
    h1=ax.scatter(dtxNoise[0:variables.Mon],ZMon[0:variables.Mon],15,ZMon[0:variables.Mon],
                  label=r'$z_{i}$')
    dPlot=np.arange(351.0)
    dPlot[0]=1
    h2=ax.plot(dPlot, variables.P-10*variables.alphai*np.log10(dPlot),color='black',linewidth=2, 
               label=r'$P-10\alpha\log_{10}(\widehat{d}_{i})$')
    h3=ax.plot(dPlot, meanP-10*meanalpha*np.log10(dPlot),'--',
               color='red', linewidth=2, 
               label=r'$\widehat{\mu}_P-10\widehat{\mu}_\alpha\log_{10}(\widehat{d}_{i})$')
    plt.xlabel(r'$\widehat{d}_{i}$ (m)')
    plt.ylabel(r'RSS [dBm]')
    handles, labels = ax.get_legend_handles_labels()
    handles=[handles[2], handles[0], handles[1]]
    labels=[labels[2], labels[0], labels[1]]
    plt.legend(handles, labels, fontsize="small")
    plt.title(r'$P=$' + str(variables.P) + r', $\mu_P=$' + str(round(meanP,2)) + 
              r', $\alpha=$' + str(variables.alphai) + r', $\mu_\alpha=$' 
              + str(round(meanalpha,2)))
    plt.axis([0, dPlot[-1], -105, 0])
    plt.grid()
    plt.savefig("variacion_potencia.eps")
    #plt.show()
    
# Figura que muestra el error cuadrático medio entre la estimación y el valor
# teórico en función de los valores de varianza de autenuación por efecto 
# "shadowing"
def representacion_final_algoritmos(flags, variables, setup, my_tplot):
    # ÚLTIMO INSTANTE DE TIEMPO
    if flags.flagGholami==1:
        mseGholami_last=setup.mseGholami[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
    
    if flags.flagWatcharapan==1:
        mseWatcharapan_last=setup.mseWatcharapan[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
    
    if flags.flagOKD==1:
        mseokd_last=setup.mseok[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
    
    if flags.flagGPstatic==1:
        if flags.GPRegression==1:   
            mseGPRstatic_last=setup.mseGPRstatic[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
        if flags.GPRegression_inducing==1:
            mseGPCstatic_last=setup.mseGPCstatic[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
        
    if flags.flagGPrecursive==1:
        mseGPrecursive_last=setup.mseGPrecursive[:,variables.Ttotal-1]/(variables.U*variables.Nexp)
    
    # MSE
    maxy=0
    
    if flags.flagGholami==1:
        h=plt.plot(variables.shadowvarSim, mseGholami_last, label='Gholami t=10')
        maxy=np.maximum(maxy, np.max(mseGholami_last))
        
    if flags.flagWatcharapan==1:
        h=plt.plot(variables.shadowvarSim, mseWatcharapan_last, label='Watcharapan t=10')
        maxy=np.maximum(maxy, np.max(mseWatcharapan_last))
    
    if flags.flagGPstatic==1:
        if flags.GPRegression==1:    
            h=plt.plot(variables.shadowvarSim, mseGPRstatic_last, label='sGP (Regression Model) t=10')
            maxy=np.maximum(maxy, np.max(mseGPRstatic_last))
        if flags.GPRegression_inducing==1:
            h=plt.plot(variables.shadowvarSim, mseGPCstatic_last, label='sGP (Sparse Model) t=10')
            maxy=np.maximum(maxy, np.max(mseGPCstatic_last))       
        
    if flags.flagGPrecursive==1:
        h=plt.plot(variables.shadowvarSim, mseGPrecursive_last, label='rGP t=10')
        maxy=np.maximum(maxy, np.max(mseGPrecursive_last))
    
    if flags.flagOKD==1:
        h=plt.plot(variables.shadowvarSim, mseokd_last, label='OKD t=10')
        maxy=np.maximum(maxy, np.max(mseokd_last))
    
    plt.xlabel('Outdoor Shadowing Variance (dB)')
    plt.ylabel('Mean Square Error (dBm)')
    plt.axis([np.min(variables.shadowvarSim)-0.5, np.max(variables.shadowvarSim)+0.5, 0, maxy+1])
    #plt.title('Last time instant')
    plt.legend(loc=2)
    
    #plt.show()
    
    # INSTANTE DE TIEMPO ESPECÍFICO
    tplot=my_tplot
    
    if flags.flagGholami==1:
        mseGholami_tplot=setup.mseGholami[:,tplot]/(variables.U*variables.Nexp)
        
    if flags.flagWatcharapan==1:
        mseWatcharapan_tplot=setup.mseWatcharapan[:,tplot]/(variables.U*variables.Nexp)
    
    if flags.flagGPstatic==1:
        if flags.GPRegression==1:
            mseGPRstatic_tplot=setup.mseGPRstatic[:,tplot]/(variables.U*variables.Nexp)
        if flags.GPRegression_inducing==1:
            mseGPCstatic_tplot=setup1.mseGPCstatic[:,tplot]/(variables.U*variables.Nexp)
        
    if flags.flagGPrecursive==1:
        mseGPrecursive_tplot=setup.mseGPrecursive[:,tplot]/(variables.U*variables.Nexp)
        
    if flags.flagOKD==1:
        mseok_tplot=setup.mseok[:,tplot]/(variables.U*variables.Nexp)
    
    # MSE
    maxy=0
    
    if flags.flagGholami==1:
        h=plt.plot(variables.shadowvarSim, mseGholami_tplot, label='Gholami t=1')
        maxy=np.maximum(maxy, np.max(mseGholami_tplot))
        
    if flags.flagWatcharapan==1:
        h=plt.plot(variables.shadowvarSim, mseWatcharapan_tplot, label='Watcharapan t=1')
        maxy=np.maximum(maxy, np.max(mseWatcharapan_tplot))
    
    if flags.flagGPstatic==1:
        if flags.GPRegression==1:   
            h=plt.plot(variables.shadowvarSim, mseGPRstatic_tplot, label='sGP (Regression Model) t=1')
            maxy=np.maximum(maxy, np.max(mseGPRstatic_tplot))
        if flags.GPRegression_inducing==1:
            h=plt.plot(variables.shadowvarSim, mseGPCstatic_tplot, label='sGP (Sparse Model) t=1')
            maxy=np.maximum(maxy, np.max(mseGPCstatic_tplot))       
        
    if flags.flagGPrecursive==1:
        h=plt.plot(variables.shadowvarSim, mseGPrecursive_tplot, label='rGP t=1')
        maxy=np.maximum(maxy, np.max(mseGPrecursive_tplot))
    
    if flags.flagOKD==1:
        h=plt.plot(variables.shadowvarSim, mseok_tplot, label='OKD t=1')
        maxy=np.maximum(maxy, np.max(mseok_tplot))
    
    plt.xlabel('Outdoor Shadowing Variance (dB)')
    plt.ylabel('Mean Square Error (dBm)')
    plt.axis([np.min(variables.shadowvarSim)-0.5, np.max(variables.shadowvarSim)+0.5, 0, maxy+1])
    #plt.title('t='+str(tplot)+' time instant')
    plt.legend(loc=2)
    plt.grid()
    plt.savefig("sGP.eps")
    plt.show()
                        


