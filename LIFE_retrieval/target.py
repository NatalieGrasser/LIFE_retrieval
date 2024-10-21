import numpy as np
import pathlib
import os

class Target:

    def __init__(self,name):
        self.name=name
        self.color1='deepskyblue' 

    def load_spectrum(self):
        filename='K2-18b-Sorg-R250.txt'
        #file=pathlib.Path(f'{self.cwd}/{self.name}/{self.name}_spectrum.txt')
        #file=pathlib.Path(f'{self.cwd}/{filename}')
        file=np.genfromtxt(filename,skip_header=1,delimiter='    ')
        self.wl=file[:,0]
        self.fl=file[:,1]/np.nanmedian(file[:,1])
        self.flerr=file[:,2]
        return self.wl,self.fl,self.flerr
        
    def get_mask_isfinite(self):
        self.n_pixels = self.fl.shape # shape (orders,detectors,pixels)
        self.mask_isfinite=np.isfinite(self.fl)
        return self.mask_isfinite
    
            
    

