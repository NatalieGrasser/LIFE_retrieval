import numpy as np
import pathlib
import os

class Target:

    def __init__(self,name):
        self.name=name
        if self.name=='test':
            self.color1='peru'
        if self.name=='Sorg1':
            self.color1='fuchsia'
        if self.name=='Sorg20':
            self.color1='tomato' 

    def load_spectrum(self):
        if self.name=='test':
            filename=f'./{self.name}/test_spectrum.txt'
            file=np.genfromtxt(filename,skip_header=0,delimiter=' ')
            self.wl=file[:,0]
            self.fl=file[:,1]/np.nanmedian(file[:,1])
            self.flerr=np.ones_like(self.fl)*0.1#file[:,2]
        elif self.name in ['Sorg1','Sorg20']:
            # weird format???
            filename=f'./{self.name}/psg_rad_{self.name}X.txt'
            file1=np.genfromtxt(filename,skip_header=13,skip_footer=1488,delimiter='  ')
            wl1=file1[:,0]
            fl1=file1[:,1]
            #flerr1=np.ones_like(fl1)*1e-5 
            file2=np.genfromtxt(filename,skip_header=424,delimiter='  ')
            wl2=file2[:,0]
            fl2=file2[:,1]
            #flerr2=file2[:,2] # why so high at the beginning?
            self.wl=np.append(wl1,wl2)
            self.fl=np.append(fl1,fl2)
            self.fl/=np.median(self.fl)
            #self.flerr=np.append(flerr1,flerr2) 
            self.flerr = np.ones_like(self.fl)*1e-5
        return self.wl,self.fl,self.flerr
        
    def get_mask_isfinite(self):
        self.n_pixels = self.fl.shape
        self.mask_isfinite=np.isfinite(self.fl)
        return self.mask_isfinite
    
            
    

