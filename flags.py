# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

class Flags:
    def __init__(self, scenario, flag_tx_est, flagGPstatic, flagGPrecursive, 
                 flagOKD, flagGholami, flagWatcharapan, flagfigs, 
                 flagdisp, flagGridFix, optimiseHyperparameters, 
                 GPRegression, GPRegression_inducing):
        self.scenario=scenario
        self.flag_tx_est=flag_tx_est 
        self.flagGPstatic=flagGPstatic
        self.flagGPrecursive=flagGPrecursive
        self.flagOKD=flagOKD
        self.flagGholami=flagGholami
        self.flagWatcharapan=flagWatcharapan
        self.flagfigs=flagfigs
        self.flagdisp=flagdisp
        self.flagGridFix=flagGridFix
        self.optimiseHyperparameters=optimiseHyperparameters
        self.GPRegression=GPRegression
        self.GPRegression_inducing=GPRegression_inducing
    
    # Selección de los flags correspondientes al escenario seleccionado
    def flag_mov_inter(self):
        if self.scenario==1:
            self.flag_intermitent=0
            self.flag_moving=0
        elif self.scenario==2:
            self.flag_intermitent=0
            self.flag_moving=1
        else:
            self.flag_intermitent=1
            self.flag_moving=1 

        

