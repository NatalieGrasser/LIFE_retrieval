import numpy as np
import os
from scipy.interpolate import CubicSpline
from astropy import constants as const
from astropy import units as u
import pandas as pd
from scipy.interpolate import interp1d

import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting
    #from LIFE_retrieval.spectrum import Spectrum, convolve_to_resolution
    from spectrum import Spectrum, convolve_to_resolution

elif getpass.getuser() == "natalie": # when testing from my laptop
    from spectrum import Spectrum, convolve_to_resolution

class pRT_spectrum:

    def __init__(self,
                 parameters,
                 data_wave,
                 target,
                 species,
                 atmosphere_object,
                 spectral_resolution=250,  
                 cloud_mode='gray',
                 contribution=False, # only for plotting atmosphere.contr_em
                 PT_type='PTgrad'):
        
        self.params=parameters
        self.data_wave=data_wave
        self.target=target
        self.species=species
        self.spectral_resolution=spectral_resolution
        self.atmosphere_object=atmosphere_object

        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)  # like in deRegt+2024
        self.PT_type=PT_type
        self.temperature = self.make_pt() #P-T profile

        self.give_absorption_opacity=None
        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution
        self.cloud_mode=cloud_mode

        self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species,self.params)
        self.MMW = self.mass_fractions['MMW']

    def read_species_info(self,species,info_key):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        if info_key == 'pRT_name':
            return species_info.loc[species,info_key]
        if info_key == 'mass':
            return species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return species_info.loc[species,info_key]
        if info_key == 'label':
            return species_info.loc[species,'mathtext_name']
    
    def free_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if line_species_i in line_species:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for line_species_i in mass_fractions.keys():
            mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary

        self.VMR_wo_H2=VMR_wo_H2[0] # same for all atmospheric layers, must be < 1
        if self.VMR_wo_H2>1: # exit if invalid params, or there will be error message
            return mass_fractions,1,1

        CO = C/O
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        return mass_fractions, CO, FeH

    def gray_cloud_opacity(self): # like in deRegt+2024
        def give_opacity(wave_micron=self.wave_micron,pressure=self.pressure):
            opa_gray_cloud = np.zeros((len(self.wave_micron),len(self.pressure))) # gray cloud = independent of wavelength
            opa_gray_cloud[:,self.pressure>10**(self.params['log_P_base_gray'])] = 0 # [bar] constant below cloud base
            # Opacity decreases with power-law above the base
            above_clouds = (self.pressure<=10**(self.params['log_P_base_gray']))
            opa_gray_cloud[:,above_clouds]=(10**(self.params['log_opa_base_gray']))*(self.pressure[above_clouds]/10**(self.params['log_P_base_gray']))**self.params['fsed_gray']
            if self.params.get('cloud_slope') is not None:
                opa_gray_cloud *= (self.wave_micron[:,None]/1)**self.params['cloud_slope']
            return opa_gray_cloud
        return give_opacity
    
    def make_spectrum(self):

        atmosphere=self.atmosphere_object
        if self.VMR_wo_H2>1: # if invalid parameters
            return np.ones_like(self.data_wave)

        if self.cloud_mode == 'gray': # Gray cloud opacity
            self.wave_micron = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
            self.give_absorption_opacity=self.gray_cloud_opacity() # fsed_gray only needed here, not in calc_flux

        atmosphere.calc_flux(self.temperature,
                        self.mass_fractions,
                        self.gravity,
                        self.MMW,
                        contribution =self.contribution,
                        give_absorption_opacity=self.give_absorption_opacity)

        wl = const.c.to(u.km/u.s).value/atmosphere.freq/1e-9 # mircons
        if np.nansum(atmosphere.flux)==0:
            print(self.params)
        flux = atmosphere.flux/np.nanmean(atmosphere.flux)
        spec = Spectrum(flux, wl)
        spec = convolve_to_resolution(spec,self.spectral_resolution)

        # Interpolate/rebin onto the data's wavelength grid
        flux = np.interp(self.data_wave, spec.wavelengths, spec)

        if self.contribution==True:
            contr_em = atmosphere.contr_em # emission contribution
            self.contr_em = np.nansum(contr_em,axis=1) # sum over all wavelengths

        return flux
            
    def make_pt(self,**kwargs): 

        if self.PT_type=='PTknot': # retrieve temperature knots
            self.T_knots = np.array([self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1'],self.params['T0']])
            self.log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=len(self.T_knots))
            sort = np.argsort(self.log_P_knots)
            self.temperature = CubicSpline(self.log_P_knots[sort],self.T_knots[sort])(np.log10(self.pressure))
        
        if self.PT_type=='PTgrad':
            self.log_P_knots = np.linspace(np.log10(np.min(self.pressure)),
                                           np.log10(np.max(self.pressure)),num=5) # 5 gradient values
            
            if 'dlnT_dlnP_knots' not in kwargs:
                self.dlnT_dlnP_knots=[]
                for i in range(5):
                    self.dlnT_dlnP_knots.append(self.params[f'dlnT_dlnP_{i}'])
            elif 'dlnT_dlnP_knots' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                self.dlnT_dlnP_knots=kwargs.get('dlnT_dlnP_knots')

            # interpolate over dlnT/dlnP gradients
            interp_func = interp1d(self.log_P_knots,self.dlnT_dlnP_knots,kind='quadratic') # for the other 50 atm layers
            dlnT_dlnP = interp_func(np.log10(self.pressure))[::-1] # reverse order, start at bottom of atm

            if 'T_base' not in kwargs:
                T_base = self.params['T0'] # T0 is free param, at bottom of atmosphere
            elif 'T_base' in kwargs: # needed for calc error on PT, upper+lower bounds passed
                T_base=kwargs.get('T_base')

            ln_P = np.log(self.pressure)[::-1]
            temperature = [T_base, ]

            # calc temperatures relative to base pressure, from bottom to top of atmosphere
            for i, ln_P_up_i in enumerate(ln_P[1:]): # start after base, T at base already defined
                ln_P_low_i = ln_P[i]
                ln_T_low_i = np.log(temperature[-1])
                # compute temperatures based on gradient
                ln_T_up_i = ln_T_low_i + (ln_P_up_i - ln_P_low_i)*dlnT_dlnP[i+1]
                temperature.append(np.exp(ln_T_up_i))
            self.temperature = temperature[::-1] # reverse order, pRT reads temps from top to bottom of atm
        
        return self.temperature