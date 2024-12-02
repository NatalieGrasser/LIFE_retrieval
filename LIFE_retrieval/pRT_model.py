import numpy as np
import os
from scipy.interpolate import CubicSpline
from astropy import constants as const
from astropy import units as u
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pickle
import getpass
if getpass.getuser() == "grasser": # when runnig from LEM
    import matplotlib
    matplotlib.use('Agg') # disable interactive plotting

class pRT_spectrum:

    def __init__(self,
                 retrieval_object,
                 spectral_resolution=250,
                 contribution=False):
        
        self.output_dir= retrieval_object.output_dir
        self.params=retrieval_object.parameters.params
        self.data_wave=retrieval_object.data_wave
        self.target=retrieval_object.target
        self.species_pRT=retrieval_object.species_pRT
        self.spectral_resolution=spectral_resolution
        self.atmosphere_object=retrieval_object.atmosphere_object

        self.n_atm_layers=retrieval_object.n_atm_layers
        self.pressure = retrieval_object.pressure
        self.PT_type=retrieval_object.PT_type
        self.temperature = self.make_pt() #P-T profile

        self.give_absorption_opacity=None
        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution
        self.cloud_mode=retrieval_object.cloud_mode

        if retrieval_object.chem=='const':
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
        elif retrieval_object.chem=='var':
            self.mass_fractions, self.CO, self.FeH = self.var_chemistry(self.species_pRT,self.params)
            self.MMW = self.mass_fractions['MMW']
            #self.VMRs = self.get_VMRs(self.mass_fractions)

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
        
    def get_VMRs(self,mass_fractions):
        species_info = pd.read_csv(os.path.join('species_info.csv'))
        VMR_dict={}
        MMW=self.MMW
        for pRT_name in mass_fractions.keys():
            if pRT_name!='MMW':
                mass=species_info.loc[species_info["pRT_name"]==pRT_name]['mass'].values[0]
                name=species_info.loc[species_info["pRT_name"]==pRT_name]['name'].values[0]
                VMR_dict[name]=mass_fractions[pRT_name]*MMW/mass
        return VMR_dict
        
    def var_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)

        #mass_fractions_list=[]
        CO_list=[]
        FeH_list=[]
        VMR_He = 0.15
        VMRs_list=[]

        for knot in range(3): # points where to retrieve abundances

            VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
            #mass_fractions = {} # Create a dictionary for all used species
            VMRs={}
            C, O, H = 0, 0, 0

            for species_i in species_info.index:
                line_species_i = self.read_species_info(species_i,'pRT_name')
                mass_i = self.read_species_info(species_i, 'mass')
                COH_i  = self.read_species_info(species_i, 'COH')

                if species_i in ['H2', 'He']:
                    continue
                if line_species_i in line_species:
                    VMR_i = 10**(params[f'log_{species_i}_{knot}'])

                    # Convert VMR to mass fraction using molecular mass number
                    #mass_fractions[line_species_i] = mass_i * VMR_i
                    VMRs[line_species_i] = VMR_i
                    VMR_wo_H2 += VMR_i

                    # Record C, O, and H bearing species for C/O and metallicity
                    C += COH_i[0] * VMR_i
                    O += COH_i[1] * VMR_i
                    H += COH_i[2] * VMR_i

            # Add the H2 and He abundances
            #mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
            VMRs['He'] = VMR_He
            H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
            self.VMR_wo_H2=VMR_wo_H2

            #MMW = 0 # Compute the mean molecular weight from all species
            #for mass_i in mass_fractions.values():
                #MMW += mass_i
            
            #for line_species_i in mass_fractions.keys():
                #mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
            #mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary

            CO = C/O
            log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
            FeH = np.log10(C/H)-log_CH_solar
            CO_list.append(CO)
            FeH_list.append(FeH)
            #mass_fractions_list.append(mass_fractions)
            VMRs_list.append(VMRs)

        #mass_fractions_interp = {}
        #for line_species_i in mass_fractions.keys():
            #mass_fracs = [mass_fractions_list[0][line_species_i],mass_fractions_list[1][line_species_i],mass_fractions_list[2][line_species_i]]
            #log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=3)
            # use linear interpolation to avoid going into negative values cubic spline did that)
            #log_mass_fracs=np.interp(np.log10(self.pressure), log_P_knots, np.log10(mass_fracs)) # interpolate for all layers
            #mass_fractions_interp[line_species_i] = 10**log_mass_fracs #np.interp(np.log10(self.pressure), log_P_knots, mass_fracs) # interpolate for all layers

        VMRs_interp = {}
        mass_fractions_interp = {}
        #for line_species_i in VMRs.keys():
        line_species.append('He')
        for species_i in species_info.index:
            #if species_i=='H2':
                #continue
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            if line_species_i in line_species and line_species_i!='H2':
                vmrs3 = [VMRs_list[0][line_species_i],VMRs_list[1][line_species_i],VMRs_list[2][line_species_i]]
                log_P_knots= np.linspace(np.log10(np.min(self.pressure)),np.log10(np.max(self.pressure)),num=3)
                # use linear interpolation to avoid going into negative values cubic spline did that)
                log_vmrs=np.interp(np.log10(self.pressure), log_P_knots, np.log10(vmrs3)) # interpolate for all layers
                VMRs_interp[line_species_i] = 10**log_vmrs #np.interp(np.log10(self.pressure), log_P_knots, mass_fracs) # interpolate for all layers
                mass_fractions_interp[line_species_i]=mass_i*VMRs_interp[line_species_i]

        vmr_layers = np.empty(self.n_atm_layers) # vmr of layers
        for l in range(self.n_atm_layers):
            vmr=0
            for key in VMRs_interp.keys():
                vmr+=VMRs_interp[key][l]
            vmr_layers[l]=vmr

        self.vmr_layers=vmr_layers
        vmr_H2=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            vmr_H2[l]=1-vmr_layers[l]
            if vmr_H2[l]<0:
                print('Invalid VMR',vmr_H2[l])
                #print(mass_fractions_interp)
                self.VMR_wo_H2=1.1
                exit_mf={}
                for key in VMRs_interp.keys():
                    exit_mf[key]=np.ones(self.n_atm_layers)*1e-12
                return exit_mf,1,1
        VMRs_interp['H2']=vmr_H2
        mass_fractions_interp['H2']=self.read_species_info('H2','mass')*VMRs_interp['H2']

        mmw_layers=np.empty(self.n_atm_layers)
        for l in range(self.n_atm_layers):
            MMW = 0 # Compute the mean molecular weight from all species for each layer
            for line_species_i in mass_fractions_interp.keys():
                #print(line_species_i, mass_fractions_interp[line_species_i][l])
                MMW += mass_fractions_interp[line_species_i][l]
            mmw_layers[l] = MMW
        mass_fractions_interp['MMW'] = mmw_layers # pRT requires MMW in mass fractions dictionary

        for line_species_i in mass_fractions_interp.keys():
            if line_species_i=='MMW':
                continue
            mass_fractions_interp[line_species_i] /= mass_fractions_interp['MMW'] # Turn the molecular masses into mass fractions
            
        #mf_layers = np.empty(self.n_atm_layers) # mass fraction of layers
        #for l in range(self.n_atm_layers):
            #mf=0
            #for key in mass_fractions_interp.keys():
                #if key!='MMW':
                    #mf+=mass_fractions_interp[key][l]
            #mf_layers[l]=mf
        
        #if any(l>1 for l in mf_layers): # exit if invalid params, or there will be error message
            
        CO = np.nanmean(CO_list)
        FeH = np.nanmean(FeH_list)
        self.VMRs=VMRs_interp
        
        return mass_fractions_interp, CO, FeH
    
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
        if np.nanmean(atmosphere.flux) in [0, np.nan, np.inf]:
            print('Invalid flux',np.nanmean(atmosphere.flux)) # cause: probably messed up PT profile
            #with open(f'{self.output_dir}/failed_pRT_obj.pickle','wb') as file: # save new results in separate dict
                #pickle.dump(self,file)
            flux= np.ones_like(wl)
        else:
            flux = atmosphere.flux/np.nanmean(atmosphere.flux)
        
        flux = self.convolve_to_resolution(wl, flux, self.spectral_resolution)

        # Interpolate/rebin onto the data's wavelength grid
        flux = np.interp(self.data_wave, wl, flux)

        if self.contribution==True:
            contr_em = atmosphere.contr_em # emission contribution
            self.contr_em = np.nansum(contr_em,axis=1) # sum over all wavelengths

        return flux
            
    def make_pt(self,**kwargs): 

        if self.PT_type=='PTknot': # retrieve temperature knots
            #self.T_knots = np.array([self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1'],self.params['T0']])
            self.T_knots = np.array([self.params['T6'],self.params['T5'],self.params['T4'],self.params['T3'],self.params['T2'],self.params['T1'],self.params['T0']])
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
    
    def convolve_to_resolution(self, wave, flux, out_res, in_res=None):
        
        in_wlen = wave # in nm
        in_flux = flux
        if isinstance(in_wlen, u.Quantity):
            in_wlen = in_wlen.to(u.nm).value
        if in_res is None:
            in_res = np.mean((in_wlen[:-1]/np.diff(in_wlen)))
        # delta lambda of resolution element is FWHM of the LSF's standard deviation:
        sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))
        spacing = np.mean(2.*np.diff(in_wlen)/(in_wlen[1:]+in_wlen[:-1]))

        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF/spacing
        result = np.tile(np.nan, in_flux.shape)
        nans = np.isnan(flux)

        result[~nans] = gaussian_filter(in_flux[~nans], sigma = sigma_LSF_gauss_filter, mode = 'reflect')
        return result
    