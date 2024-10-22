import numpy as np
import pathlib
import os

class Target:

    def __init__(self,name):
        self.name=name
        self.color1='tomato' 

    def load_spectrum(self):
        filename='spec_emiss_0.txt'
        file=np.genfromtxt(filename,skip_header=0,delimiter=' ')
        self.wl=file[:,0]
        self.fl=file[:,1]/np.nanmedian(file[:,1])
        self.flerr=np.ones_like(self.fl)*0.1#file[:,2]
        return self.wl,self.fl,self.flerr
        
    def get_mask_isfinite(self):
        self.n_pixels = self.fl.shape # shape (orders,detectors,pixels)
        self.mask_isfinite=np.isfinite(self.fl)
        return self.mask_isfinite
    
            
    

