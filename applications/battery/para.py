from dataclasses import dataclass
from matlab_fns import calcUoc, calcKappa

import numpy as onp

@dataclass
class param_sets_macro:
    
    dt:float # time step size
    
    def __post_init__(self):
        
        # ---- physical constants ----
        
        self.F = 96485.33289 # Faraday's constant (C/mol)
        self.R = 8.31447 # Gas constant
        self.T = 298.15 # Absolute temperature 25 celsius
        
        self.phi_ref = self.R * self.T / self.F;
        
        # ---- interfacial kinetics ----
        
        self.k_s = [1e-10, 3e-11] # m^2.5 mol^-0.5 s^-1
        
        # maximum conc in cathode active material (mol/m^3)
        self.c_max = [2.4983e+04, 5.1218e+04]
        
        # ---- initial conditions ----
        
        # Initial concentration in electrolyte (mol/m^3)
        self.c0_el = 1.0e3 
        # Initial potential (V)
        self.phi0_el = 0.
        
        self.c0_an = 19624.; # 0.5164 * self.c_max(1);
        self.phi0_an = calcUoc(self.c0_an, 1, self)
        
        self.c0_ca = 20046.
        self.phi0_ca = calcUoc(self.c0_ca, 2, self)
    
        
        self.l_ref = 1.0e-6 # Characteristic length = length of separator (m)

        self.eps_inc_an = 0.6 # Volume fraction of active particles in anode
        self.eps_mat_an = 0.3 # Volume fraction of electrolyte in anode
        self.eps_fil_an = 0.1 # Volume fraction of filler in anode
        
        self.eps_inc_ca = 0.5 # Volume fraction of active particles in cathode
        self.eps_mat_ca = 0.3 # Volume fraction of electrolyte in cathode
        self.eps_fil_ca = 0.2 # Volume fraction of filler in cathode
        
        self.eps_mat_se = 1.0 # Volume fraction of electrolyte in separator
        
        # particle radius in anode
        self.r_an = 10e-6 # (m)
        
        # particle radius in cathode
        self.r_ca = 10e-6 # (m)
        
        self.alpha = 1.5
        
        # ---- transport properties for electrolyte ----
        
        # ionic conductivity
        self.kappa = calcKappa(self.c0_el)
        self.ka_ref = self.kappa
        
        # diffusivity
        self.df = 2.7877e-10 # 5.34e-10
        self.df_ref = self.df
        
        # transference number
        self.tp = 0.4
        
        # coefficient for conc time diff term
        self.dtco_mat = 1 / self.dt * self.l_ref**2 / self.df_ref
        
        # coefficient for current flux term
        self.i_e = self.l_ref / self.ka_ref
        # coefficient for species flux term
        self.q_e = self.l_ref / self.df_ref / self.F
        
        # coefficient for current source term
        self.sour_p = (self.l_ref)**2 / self.ka_ref * self.F
        # coefficient for species source term
        self.sour_c = (self.l_ref)**2 / self.df_ref * (1 - self.tp)
        
        # ---- transport properties for active material in anode ---- 
        
        # active material conductivity
        self.sigan = 100. # (s/m)
        self.sigan_ref = self.sigan
        
        # diffusivity
        self.dan = 3.9e-14 # (m^2/s)
        self.dan_ref = self.dan
        
        self.dtco_an = 1 / self.dt * self.l_ref**2 / self.dan_ref
        
        self.i_an = self.l_ref / self.sigan_ref
        
        self.sour_an = (self.l_ref)**2 / self.sigan_ref * self.F
        
        # ---- transport properties for active material in cathode ---- 
        
        # active material conductivity
        self.sigca = 10. # (s/m)
        self.sigca_ref = self.sigca
        
        # diffusivity
        self.dca = 1e-13 # (m^2/s)
        self.dca_ref = self.dca
        
        self.dtco_ca = 1 / self.dt * self.l_ref**2 / self.dca_ref
        
        self.i_ca = self.l_ref / self.sigca_ref
        
        self.sour_ca = (self.l_ref)**2 / self.sigca_ref * self.F
        
        # ---- applied external current ----
        
        self.I_bc = 30. # (A/m^2)
        

@dataclass
class param_sets_micro:
    
    dt: float
    r_an: float
    r_ca: float
    
    def __post_init__(self):
        
        # ---- physical constants ----
        
        self.F = 96485.33289     # Faraday's constant (C/mol)
        self.R = 8.31447         # Gas constant
        self.T = 298.15          # Absolute temperature 25 cels
        
        
        self.l_ref_an = self.r_an
        self.l_ref_ca = self.r_ca
        
        
        # ---- anode ----
        
        df_an = 3.9e-14 # m^2/s
        df_an_ref = df_an
        dtco_an = 1 / self.dt * self.l_ref_an**2 / df_an_ref
        q_an = 4 * onp.pi * self.r_an**2 / df_an_ref / self.l_ref_an
        
        self.ds_an = df_an;
        self.ds_ref_an = df_an_ref;
        self.dtco_inc_an = dtco_an;
        self.q_bc_an = q_an;
        
        
        # ---- cathode ----
        
        df_ca = 1e-13   # m^2/s
        df_ca_ref = df_ca
        dtco_ca = 1 / self.dt * self.l_ref_ca**2 / df_ca_ref
        q_ca = 4 * onp.pi * self.r_ca**2 / df_ca_ref / self.l_ref_ca
        
        self.ds_ca = df_ca;
        self.ds_ref_ca = df_ca_ref;
        self.dtco_inc_ca = dtco_ca;
        self.q_bc_ca = q_ca;


@dataclass
class data_sets:
    id:str
    
