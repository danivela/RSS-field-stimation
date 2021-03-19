# Autor: Daniel Vela Calderón
# Trabajo Fin de Máster

import math

class Variables:
    def __init__(self, shadowvarSim, Nexp, Ttotal, P, U, alphai, sigma_u, Dcorr, 
                 varnoise, forgetting_factor, rateoff, var_movement, noise_sigmau, 
                 noise_varnoise):
        self.shadowvarSim=shadowvarSim
        self.Nexp=Nexp
        self.Ttotal=Ttotal
        self.P=P
        self.U=U
        self.alphai=alphai
        self.sigma_u=sigma_u
        # Obtenida a partir de la expresión bajo la fórmula 9 
        self.sigma_d=self.sigma_u*math.log(10)/(10*self.alphai)
        self.Dcorr=Dcorr
        self.varnoise=varnoise
        self.forgetting_factor=forgetting_factor
        self.rateoff=rateoff
        self.var_movement=var_movement
        self.noise_sigmau=noise_sigmau
        self.noise_varnoise=noise_varnoise
        self.M=0
        self.N=0
        self.Mon=0
        self.Moff=0

