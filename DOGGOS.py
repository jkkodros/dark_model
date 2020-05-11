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
    
    def __init__(self, NO_i, NO2_i, O3_i, phenol_i, cresol_i):
        # Instance variables of initial values for model
        self.NO_i = NO_i
        self.NO2_i = NO2_i
        self.NO3_i = NO3_i
        self.phenol_i = phenol_i
        self.cresol_i = cresol_i
        
        
    
