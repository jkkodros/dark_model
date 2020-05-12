# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:27:31 2020

Simple dark chemistry box model

@author: J Kodros
"""

import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate
from functools import partial
import matplotlib.pyplot as plt
import sys

class DOGGOS:
    '''Class decleration for simple dark chemistry box model: Dark
    Oxidation Of Gas and Grouped Organic Species'''
    
    def __init__(self, NO_i_ppb, NO2_i_ppb, O3_i_ppb, NO3_i_ppb, phenol_i_ppb, 
                 cresol_i_ppb, simulation_time, temperature=19.0, RH=0.5,
                 initialize_time=np.array([None]), experiment_time=np.array([None])):
        # Instance variables of initial values for model
        self.NO = [NO_i_ppb]
        self.NO2 = [NO2_i_ppb]
        self.O3 = [O3_i_ppb]
        self.NO3 = [NO3_i_ppb]
        self.N2O5 = [0.0]
        self.phenol = [phenol_i_ppb]
        self.cresol = [cresol_i_ppb]
        self.product1 = [0.0]
        self.product2 = [0.0]
        self.product3 = [0.0]
        self.HNO3 = [0.0]
        self.NA = [0.0]
        self.temperature = [temperature]
        self.RH = [RH]
        self.simulation_time = np.array(simulation_time)
        self.initialize_time = np.array(initialize_time)
        
        if initialize_time.any():
            finterp_NO2 = interpolate.interp1d(initialize_time, NO2_i_ppb)
            NO2 = finterp_NO2(self.simulation_time[self.simulation_time 
                                                   <= self.initialize_time[-1]])
            self.NO2 = NO2.tolist()
            
            finterp_O3 = interpolate.interp1d(initialize_time, O3_i_ppb)
            O3 = finterp_O3(self.simulation_time[self.simulation_time 
                                                   <= self.initialize_time[-1]])
            self.O3 = O3.tolist()
            
        if experiment_time.any():
            finterp_T = interpolate.interp1d(experiment_time, temperature)
            temperature = finterp_T(self.simulation_time)
            self.temperature = temperature.tolist()
            
            finterp_RH = interpolate.interp1d(experiment_time, RH)
            RH = finterp_RH(self.simulation_time)
            self.RH = RH.tolist()
            
        
        self.R1 = [0.0]
        self.R2 = [0.0]
        self.R3 = [0.0]
        self.R4 = [0.0]
        self.R5 = [0.0]
        self.R6 = [0.0]
        self.R7 = [0.0]
        self.R8 = [0.0]
        self.R9 = [0.0]
        self.R10 = [0.0]
        self.R11 = [0.0]
        self.R12 = [0.0]
        self.R13 = [0.0]
        self.R14 = [0.0]
        self.R15 = [0.0]
        self.R16 = [0.0]
        self.R17 = [0.0]

        
    def run_model(self):
        # Tracers
        y0_ppb = [self.NO[0],
                  self.NO2[0],
                  self.O3[0],
                  self.NO3[0],
                  self.N2O5[0],
                  self.phenol[0], 
                  self.cresol[0],
                  self.product1[0],
                  self.product2[0],
                  self.product3[0],
                  self.HNO3[0],
                  self.NA[0]]
        
        y0 = [convert_ppb_to_molec_cm3(val) for val in y0_ppb]
        
        # Time step
        for i in range(1, len(self.simulation_time)):
            # span for next time step
            tspan = [self.simulation_time[i-1], self.simulation_time[i]]
            
            # Reaction budgets
            R = reactions(y0, tspan[0], self.temperature[i], forward=False)
            
            R_ppb = [convert_molec_cm3_to_ppb(val) for val in R]
            self.R1.append(R_ppb[0])
            self.R2.append(R_ppb[1])
            self.R3.append(R_ppb[2])
            self.R4.append(R_ppb[3])
            self.R5.append(R_ppb[4])
            self.R6.append(R_ppb[5])
            self.R7.append(R_ppb[6])
            self.R8.append(R_ppb[7])
            self.R9.append(R_ppb[8])
            self.R10.append(R_ppb[9])
            self.R11.append(R_ppb[10])
            self.R12.append(R_ppb[11])
            self.R13.append(R_ppb[12])
            self.R14.append(R_ppb[13])
            self.R15.append(R_ppb[14])
            self.R16.append(R_ppb[15])
            self.R17.append(R_ppb[16])
            
            # solve for next step
            forward=True
            y = integrate.odeint(reactions, y0, tspan, 
                                 args=(self.temperature[i], forward))
                        
            y_current = y[1]
            y_ppb = [convert_molec_cm3_to_ppb(val) for val in y_current]
            
            if self.initialize_time.any() and tspan[1] <= self.initialize_time[-1]:
                self.NO.append(y_ppb[0])        
                self.NO3.append(y_ppb[3])
                self.N2O5.append(y_ppb[4])
                self.phenol.append(y_ppb[5])
                self.cresol.append(y_ppb[6])
                self.product1.append(y_ppb[7])
                self.product2.append(y_ppb[8])
                self.product3.append(y_ppb[9])
                self.HNO3.append(y_ppb[10])
                self.NA.append(y_ppb[11])
                
                y0 = y_current
                y0[1] = convert_ppb_to_molec_cm3(self.NO2[i])
                y0[2] = convert_ppb_to_molec_cm3(self.O3[i])
            
            else:
                self.NO.append(y_ppb[0])
                self.NO2.append(y_ppb[1])
                self.O3.append(y_ppb[2])        
                self.NO3.append(y_ppb[3])
                self.N2O5.append(y_ppb[4])
                self.phenol.append(y_ppb[5])
                self.cresol.append(y_ppb[6])
                self.product1.append(y_ppb[7])
                self.product2.append(y_ppb[8])
                self.product3.append(y_ppb[9])
                self.HNO3.append(y_ppb[10])
                self.NA.append(y_ppb[11])        
                        
                # next initial condition
                y0 = y_current

    def plot_main_results(self, time_exp=[None], O3_exp=[None], NO2_exp=[None],
                          phenol_exp=[None], cresol_exp=[None]):
        fig, ax = plt.subplots(2,2, sharex=True, figsize=(10,7))
        
        time_exp = np.array(time_exp)
        O3_exp = np.array(O3_exp)
        NO2_exp = np.array(NO2_exp)
        phenol_exp = np.array(phenol_exp)
        cresol_exp = np.array(cresol_exp)
        
        
        if time_exp.any() and O3_exp.any():
            ax[0,0].plot(time_exp, O3_exp, 'o', color='red')
        ax[0,0].plot(self.simulation_time/3600., self.O3, color='red')
        ax[0,0].set_ylabel('O$_{3}$ [ppb]')
        ax[0,0].set_xlim(0, self.simulation_time[-1]/3600.)
        ax[0,0].set_ylim(0)
        
        if time_exp.any() and NO2_exp.any():
            ax[0,1].plot(time_exp, NO2_exp, 'o', color='green')
        ax[0,1].plot(self.simulation_time/3600., self.NO2, color='green')
        ax[0,1].set_ylabel('NO$_{2}$ [ppb]')
        ax[0,1].set_ylim(0)
        
        ax[1,1].plot(self.simulation_time/3600., np.array(self.NO3)*1e3, 
          color='C3')
        ax[1,1].plot(self.simulation_time/3600., self.N2O5, color='C4')
        ax[1,1].set_xlabel('Time [hr]')
        ax[1,1].set_ylabel('NO$_{3}$ [ppt] and N$_{2}$O$_{5}$ [ppb]')
        ax[1,1].set_ylim(0)
        ax[1,1].legend(['NO$_{3}$ [ppt]', 'N$_{2}$O$_{5}$ [ppb]'])

        if time_exp.any() and phenol_exp.any():
            ax[1,0].plot(time_exp, phenol_exp, 'o', color='C0')
        if time_exp.any() and phenol_exp.any():
            ax[1,0].plot(time_exp, cresol_exp, 'o', color='C1')
        ax[1,0].plot(self.simulation_time/3600., self.phenol, color='C0')
        ax[1,0].plot(self.simulation_time/3600., self.cresol, color='C1')
        ax[1,0].legend(['Phenol', 'Cresol'])
        ax[1,0].set_xlabel('Time [hr]')
        ax[1,0].set_ylabel('VOC [ppb]')
        ax[1,0].set_ylim(0, max(self.phenol))
        
        fig.tight_layout()
        plt.show()
        return fig, ax
    
    def get_NO3_budget(self):
        # Get fractional NO3 reaction budget
        R_phenol = np.array(self.R8)
        R_cresol = np.array(self.R10)
        R_products = np.array(self.R11) + np.array(self.R13)
        R_NOx = np.array(self.R3) + np.array(self.R6) + np.array(self.R7)
        R_NO3_rates = [R_phenol, R_cresol, R_products, R_NOx]
        
        R_NO3 = [integrate.simps(R, 
                                 x=self.simulation_time) for R in R_NO3_rates]

        R_NO3 = R_NO3[:]/sum(R_NO3)
            
        reaction_labels = ['Phenol',
                           'Cresol',
                           'Products',
                           'NO$_{x}$']
        
        return R_NO3, reaction_labels
    
    def plot_NO3_budget(self):
        # Plot NO3 reaction budget
        R_NO3, reaction_labels = self.get_NO3_budget()
        
        fig, ax = plt.subplots(figsize=(6,6))
        ax.bar(np.arange(len(R_NO3)), R_NO3)
        ax.set_xticks(np.arange(len(R_NO3)))
        ax.set_xticklabels(reaction_labels, rotation=45, fontsize=16)
        ax.set_ylabel('Percent of Reactions')
        ax.set_title('NO$_{3}$ Reaction Budget')
        fig.tight_layout()
        return fig, ax
            
###############################################################################
### Box model reactions
###############################################################################
def reactions(y, t, temperature, forward=True):
    # This function defines the differential equations (the reactions)
    # of the model (dC/dt where C is a compound)
    # R1 NO + O3 --> NO2  (k1) 
    # R2 NO2 + O3 --> NO3 (k2)
    # R3 NO3 + NO2 --> N2O5 (k3)
    # R4 N2O5 --> NO3 + NO2 (k4 = k3/K)
    # R5 N2O5 --> HNO3 --> NIT (k5)
    # R6 NO2 + NO3 --> NO + NO2  (k6)
    # R7 NO + NO3 --> NO2 + NO2 (k7)
    # R8 NO3 + phenol --> product1
    # R9 O3 + phenol --> product1
    # R10 NO3 + cresol --> product1
    # R11 NO3 + product1 --> product2
    # R12 O3 + product1 --> product2
    # R13 NO3 + product2 --> product3
    
    T_K = temperature + 273.15
    
    # Rate constants 
    k1 = 1.4E-12*np.exp(-1310./(T_K))        
    k2 = 1.3E-13*np.exp(-2470./(T_K))        
    k3 = 9.44E-10 * np.exp(-2509./(T_K))    
    K = 2.0E-27 * np.exp(11000./(T_K))  
    k4 = k3/K      
    k5 = 2.5E-6
    k6 = 4.5E-14*np.exp(-1260./(T_K))
    k7 = 2.6E-11*np.exp(110./(T_K)) 
    k8 = 3.8E-13  
    k9 = 1E-18 
    k10 = 1.4E-11  
    k11 = 2.3E-12
    k12 = 2.86E-13 
    k13 = 1e-12
    k14 = 6.0E-6
    k15 = 1.2E-5

    # Concentrations of gas phase species 
    NO = y[0]
    NO2 = y[1]
    O3 = y[2]
    NO3 = y[3]
    N2O5 = y[4]
    phenol = y[5]
    cresol = y[6]
    product1 = y[7]
    product2 = y[8]
    product3 = y[9]
    HNO3 = y[10]
    NA = y[11]
    
    #R1 NO + O3 --> NO2
    if NO >= 0 and O3 >= 0:
        R1 = NO * O3 * k1 
    else:
        R1 = 0.0       
    # R2 NO2 + O3 --> NO3
    if NO2 >= 0 and O3 >= 0:
        R2 = NO2 * O3 * k2
    else:
        R2 = 0.0 
    # R3 NO3 + NO2 --> N2O5
    if NO3 >= 0 and NO2 >= 0:
        R3 = NO3 * NO2 * k3
    else:
        R3 = 0.0
    #R4 N2O5 --> NO3 + NO2
    if N2O5 >= 0:
        R4 = N2O5 * k4
    else:
        R4 = 0
    # R5 N2O5 --> HNO3
    if N2O5 >= 0.0:
        R5 = N2O5 * k5
    else:
        R5 = 0.0
    # R6 NO2 + NO3 --> NO + NO2
    if NO2 >= 0.0 and NO3 >= 0.0:
        R6 = NO2*NO3*k6
    else:
        R6 = 0.0
    # R7 NO + NO3 --> NO2 + NO2
    if NO >= 0.0 and NO3 >= 0.0:
        R7 = NO*NO3*k7
    else:
        R7 = 0.0
    #R8 NO3 + phenol --> product1        
    if phenol >= 0.0 and NO3 >= 0.0:
        R8 = phenol * NO3 * k8
    else:
        R8 = 0.0
    #R9 O3 + phenol --> product1
    if phenol >= 0.0 and O3 >= 0.0:
        R9 = phenol * O3 * k9
    else:
        R9 = 0.0
    # R10 NO3 + cresol --> product1 
    if cresol >= 0.0 and NO3 >= 0.0:
        R10 = cresol * NO3 * k10
    else:
        R10 = 0.0 
    # R11 NO3 + product1 --> product2
    if NO3 >= 0 and product1 >= 0:
        R11 = product1*NO3*k11
    else:
        R11 = 0.0
    # R12 O3 + product1 --> product2 
    if product1 >= 0.0 and O3>=0.0:
        R12 = product1*O3*k12
    else:
        R12=0.0
    # R13 NO3 + product2 --> product3 
    if product2 >= 0 and NO3 >= 0:
        R13 = product2*NO3* k13
    else:
        R13 = 0.0

    ### Aerosol nitrate attempts
    #R14 HNO3 --> NIT
    if HNO3 >= 0.0:
        #R14 = HNO3 * k14
        R14 = 0.0
    else:
        R14 = 0.0
    # R15 NIT --> wall loss
    if NA >=0.0:
        R15 = NA * k15
    else:
        R15 = 0.0
        
    # Aux mechanisms
    if O3 >= 0.0:
        Raux1 = O3*1.02E-6
    else:
        Raux1 = 0.0 
    # Aux mechanisms
    if NO2 >= 0.0:
        Raux2 = NO2*8.12E-7
    else:
        Raux2 = 0.0

    # time differentials
    dNO_dt = (R6) - (R1 + R7) 
    dNO2_dt = (R1 + R4 + R6 + 2*R7 + R11) - (R2 + R3 + R6 + Raux2)
    dO3_dt = -1.0*(R1 + R2  + R9 + R12 + Raux1)
    dNO3_dt = (R2 + R4) - (R3 + R8 + R10 + R6 + R7 + R11 + R13)
    dN2O5_dt = R3 - (R4 + R5)
    dPhenol_dt = -1.0*(R8 + R9)
    dCresol_dt = -1.0*(R10)
    dProduct1_dt = (R10 + R8 + R9) - (R12 + R11)
    dProduct2_dt = (R11 + R12) - (R13)
    dProduct3_dt = R13
    dHNO3_dt = (R8*0.7 + R10*0.5) - (R14)
    dNA_dt = (R14) - (R15) +2*R5
    
    if forward:
        return [dNO_dt, dNO2_dt, dO3_dt, dNO3_dt, dN2O5_dt, dPhenol_dt, dCresol_dt,
                dProduct1_dt, dProduct2_dt, dProduct3_dt, dHNO3_dt, dNA_dt]
    else:
        return [R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14,
                R15, Raux1, Raux2]        
        
###############################################################################
### Other helpful routines
###############################################################################
def convert_ppb_to_molec_cm3(value, P=1.01E5, T=298):
    # convert gas phase ppb to ug m-3
    R = 8.3014
    Av = 6.02E23
    new_val = value * 1e-9 * (P/(R*T)) * (Av) * 1e-6 
    return new_val

def convert_molec_cm3_to_ppb(val, P=1.01E5, T=298):
    R = 8.3014
    Av = 6.02E23
    new_val = val * 1e9 * ((R*T)/P) * (Av)**(-1) *1e6
    return new_val