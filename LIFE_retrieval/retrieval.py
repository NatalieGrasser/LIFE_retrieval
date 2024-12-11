import os
import numpy as np
import pymultinest
import pathlib
import pickle
from petitRADTRANS import Radtrans
import pandas as pd
import matplotlib.pyplot as plt
import getpass

if getpass.getuser() == "grasser": # when running from LEM
    os.environ['pRT_input_data_path'] = "/net/lem/data2/pRT_input_data"
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
from pRT_model import pRT_spectrum
import figures as figs
from covariance import *
from log_likelihood import *


class Retrieval:

    def __init__(self,target,parameters,output_name,N_live_points=400,
                 evidence_tolerance=0.5,PT_type='PTgrad',chem='const'):
        
        self.target=target
        self.N_live_points=N_live_points
        self.evidence_tolerance=evidence_tolerance
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans
        self.parameters=parameters
        self.chem=chem
        self.species_pRT, self.species_hill =self.get_species(param_dict=self.parameters.params)

        self.n_pixels=len(self.data_wave)
        self.n_params = len(parameters.free_params)
        self.output_name=output_name
        self.cwd = os.getcwd()
        self.output_dir = pathlib.Path(f'{self.cwd}/{self.target.name}/{self.output_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cloud_mode=None # change later
        self.PT_type=PT_type
        self.n_atm_layers=50
        self.pressure = np.logspace(-7,0,self.n_atm_layers) # equchem table P only until -6

        mask = self.mask_isfinite # only finite pixels
        self.Cov = Covariance(err=self.data_err[mask]) # use simple diagonal covariance matrix
        self.LogLike = LogLikelihood(retrieval_object=self,scale_flux=True,scale_err=True)

        # load atmosphere object here and not in likelihood/pRT_model to make it faster
        # redo atmosphere object when introdocuing new species
        self.atmosphere_object=self.get_atmosphere_object()
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'

        # will be updated, but needed as None until then
        self.bestfit_params=None 
        self.posterior = None
        self.params_dict=None
        self.color1=target.color1

    def get_species(self,param_dict): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        self.species_names=[]
        if self.chem=='equ':
            # species in chemical equilibrium table
            self.species_names = ['log_H2O','log_CO','log_CO2','log_CH4','log_NH3','log_HCN',
                                  'log_H2S','log_C2H2','log_C2H4','log_SO2','log_OCS','log_CS2']
        for par in param_dict:
            if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                if par in ['log_g','log_P_base_gray','log_opa_base_gray']: # skip
                    continue
                if self.chem in 'const':
                    self.species_names.append(par)
                elif self.chem=='var' and '_1' in par:
                    self.species_names.append(par[:-2])
                elif self.chem=='equ':
                    self.species_names.append(par)
        species=[]
        hill=[] # hill notation
        for chemspec in self.species_names:
            species.append(species_info.loc[chemspec[4:],'pRT_name'])
            hill.append(species_info.loc[chemspec[4:],'Hill_notation'])
        return species, hill

    def get_atmosphere_object(self):

        file=pathlib.Path(f'atmosphere_object.pickle')
        if file.exists():
            with open(file,'rb') as file:
                atmosphere_object=pickle.load(file)
                return atmosphere_object
        else:
            wl_pad=1e-2 # wavelength padding because spectrum is not RV-shifted yet
            wlmin=np.min(self.data_wave)-wl_pad
            wlmax=np.max(self.data_wave)+wl_pad
            wlen_range=np.array([wlmin,wlmax]) # already in microns for pRT

            atmosphere_object = Radtrans(line_species=self.species_pRT,
                                rayleigh_species = ['H2', 'He'],
                                continuum_opacities = ['H2-H2', 'H2-He'],
                                wlen_bords_micron=wlen_range, 
                                mode='c-k')
            
            atmosphere_object.setup_opa_structure(self.pressure)
            with open(file,'wb') as file:
                pickle.dump(atmosphere_object,file)
            return atmosphere_object

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(self)
        self.model_flux=self.model_object.make_spectrum()
        self.Cov(self.parameters.params)
        ln_L = self.LogLike(self.model_flux, self.Cov) # retrieve log-likelihood

        if False:
            plt.plot(self.data_wave,self.data_flux)
            plt.plot(self.data_wave,self.model_flux,alpha=0.7)

        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=True):
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'./{self.target.name}/{self.output_name}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True, sampling_efficiency = 0.5,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=100)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        self.posterior = posterior[:,:-2] # remove last 2 columns
        self.params_dict,self.model_flux=self.get_params_and_spectrum()
        figs.summary_plot(self)
        if self.chem=='var':
            figs.VMR_plot(self)
     
    def PMN_analyse(self):
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params, 
                                        outputfiles_basename=f'{self.output_dir}/{self.prefix}')  # set up analyzer object
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] # shape 
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.bestfit_params = np.array(stats['modes'][0]['maximum a posterior']) # read params of best-fitting model, highest likelihood
        if self.prefix=='pmn_':
            self.lnZ = stats['nested importance sampling global log-evidence']
        else: # when doing exclusion retrievals
            self.lnZ_ex = stats['nested importance sampling global log-evidence']

    def get_quantiles(self,posterior,flat=False):
        if flat==False: # input entire posterior of all retrieved parameters
            quantiles = np.array([np.percentile(posterior[:,i], [16.0,50.0,84.0], axis=-1) for i in range(posterior.shape[1])])
            medians=quantiles[:,1] # median of all params
            plus_err=quantiles[:,2]-medians # +error
            minus_err=quantiles[:,0]-medians # -error
        else: # input only one posterior
            quantiles = np.array([np.percentile(posterior, [16.0,50.0,84.0])])
            medians=quantiles[:,1] # median
            plus_err=quantiles[:,2]-medians # +error
            minus_err=quantiles[:,0]-medians # -error
        return medians,minus_err,plus_err

    def get_params_and_spectrum(self): 

        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists():
            with open(final_dict,'rb') as file:
                self.params_dict=pickle.load(file)

            for key in self.parameters.param_keys:
                self.parameters.params[key]=self.params_dict[key] # set parameters to retrieved values

            # create final spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux0=self.model_object.make_spectrum()
            self.model_flux=np.zeros_like(self.model_flux0)
            self.summed_contr=self.model_object.contr_em
            phi=self.params_dict['phi']
            self.model_flux=phi*self.model_flux0 # scale model accordingly

            # get errors and save them in final params dict
            self.get_errors()

        else:
        
            # make dict of constant params + evaluated params + their errors
            self.params_dict=self.parameters.constant_params.copy() # initialize dict with constant params
            medians,minus_err,plus_err=self.get_quantiles(self.posterior)

            for i,key in enumerate(self.parameters.param_keys):
                self.params_dict[key]=medians[i] # add median of evaluated params (more robust)
                #self.params_dict[f'{key}_bf']=self.bestfit_params[i] # bestfit params with highest lnL (can differ from median, not as robust)

            for i,key in enumerate(self.parameters.param_keys): # avoid messing up order of free params
                self.params_dict[f'{key}_err']=(minus_err[i],plus_err[i]) # add errors of evaluated params

            # create model spectrum
            self.model_object=pRT_spectrum(self,contribution=True)
            self.model_flux=self.model_object.make_spectrum()
            self.summed_contr=self.model_object.contr_em
            self.get_errors() # for temperature, C/O and [C/H]

            # get scaling parameters phi and s2 of bestfit model through likelihood
            self.log_likelihood = self.LogLike(self.model_flux, self.Cov)
            self.params_dict['phi']=self.LogLike.phi
            self.params_dict['s2']=self.LogLike.s2
            if self.callback_label=='final_':
                self.params_dict['chi2']=self.LogLike.chi2_0_red # save reduced chi^2 of fiducial model
                self.params_dict['lnZ']=self.lnZ # save lnZ of fiducial model

            phi=self.params_dict['phi']
            self.model_flux=phi*self.model_flux # scale model accordingly
            spectrum=np.full(shape=(self.n_pixels,2),fill_value=np.nan)
            spectrum[:,0]=self.data_wave
            spectrum[:,1]=self.model_flux

            if self.callback_label=='final_': # only save if final
                with open(f'{self.output_dir}/params_dict.pickle','wb') as file:
                    pickle.dump(self.params_dict,file)
                np.savetxt(f'{self.output_dir}/bestfit_spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.params_dict,self.model_flux

    def get_errors(self): # can only be run after self.evaluate()

        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)

        ratios=pathlib.Path(f'{self.output_dir}/CO_CH_dist.npy')
        temp_dist=pathlib.Path(f'{self.output_dir}/temperature_dist.npy')
        VMR_dict=pathlib.Path(f'{self.output_dir}/VMR_dict.pickle')

        if ratios.exists() and temp_dist.exists() and self.chem=='const':
            self.CO_CH_dist=np.load(ratios)
            self.temp_dist=np.load(temp_dist)

        elif ratios.exists() and temp_dist.exists() and VMR_dict.exists() and self.chem in ['var','equ']:
            self.CO_CH_dist=np.load(ratios)
            self.temp_dist=np.load(temp_dist)
            with open(VMR_dict,'rb') as file:
                self.VMR_dict=pickle.load(file)

        else:
            CO_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
            CH_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
            temperature_distribution=[] # for each of the n_atm_layers
            VMRs=[]
            for j,sample in enumerate(self.posterior):
                # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
                cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
                self.parameters(cube)
                model_object=pRT_spectrum(self)
                CO_distribution[j]=model_object.CO
                CH_distribution[j]=model_object.FeH
                temperature_distribution.append(model_object.temperature)
                if self.chem in ['var','equ']:
                    VMRs.append(model_object.VMRs)
            self.CO_CH_dist=np.vstack([CO_distribution,CH_distribution]).T
            self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)

            median,minus_err,plus_err=self.get_quantiles(CO_distribution,flat=True)
            self.params_dict['C/O']=median
            self.params_dict['C/O_err']=(minus_err,plus_err)

            median,minus_err,plus_err=self.get_quantiles(CH_distribution,flat=True)
            self.params_dict['C/H']=median
            self.params_dict['C/H_err']=(minus_err,plus_err)

            if self.chem in ['var','equ']:
                self.VMR_dict={}
                for molec in VMRs[0].keys():
                    vmr_list=[]
                    for i in range(len(self.posterior)):
                        vmr_list.append(VMRs[i][molec])
                    self.VMR_dict[molec]=vmr_list # reformat to make it easier to work with

            if self.callback_label=='final_' and getpass.getuser() == "grasser": # when running from LEM
                np.save(f'{self.output_dir}/CO_CH_dist.npy',self.CO_CH_dist)
                np.save(f'{self.output_dir}/temperature_dist.npy',self.temp_dist)
                if self.chem in ['var','equ']:
                    with open(f'{self.output_dir}/VMR_dict.pickle','wb') as file:
                        pickle.dump(self.VMR_dict,file)

    def evaluate(self,callback_label='final_',makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux=self.get_params_and_spectrum() # all params: constant + free + scaling phi + s2
        if makefigs:
            if callback_label=='final_':
                figs.make_all_plots(self,split_corner=True)
            else:
                figs.summary_plot(self)

    def bayes_evidence(self,molecules,retrieval_output_dir):

        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(molecules, list)==False:
            molecules=[molecules] # if only one, make list so that it works in for loop

        for molecule in molecules: # exclude molecule from retrieval

            finish=pathlib.Path(f'{self.output_dir}/final_wo{molecule}_posterior.npy')
            if finish.exists():
                print(f'\n ----------------- Evidence retrieval for {molecule} already done ----------------- \n')
                setback_prior=False
            else:
                print(f'\n ----------------- Starting evidence retrieval for {molecule} ----------------- \n')
                setback_prior=True
                original_prior=self.parameters.param_priors[f'log_{molecule}']
                self.parameters.param_priors[f'log_{molecule}']=[-15,-14] # exclude from retrieval
                self.callback_label=f'live_wo{molecule}_'
                self.prefix=f'pmn_wo{molecule}_' 
                self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance,resume=True)
            
            self.callback_label=f'final_wo{molecule}_'
            self.evaluate(callback_label=self.callback_label) # gets self.lnZ_ex
            ex_model=pRT_spectrum(self).make_spectrum()      
            lnL = self.LogLike(ex_model, self.Cov) # call function to generate chi2
            chi2_ex = self.LogLike.chi2_0_red # reduced chi^2
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'sigma_{molecule}=',sigma)
            self.evidence_dict[f'lnBm_{molecule}']=lnB
            self.evidence_dict[f'sigma_{molecule}']=sigma
            self.evidence_dict[f'chi2_wo_{molecule}']=chi2_ex
            print('bayes_dict=',self.evidence_dict)  
            with open(f'{retrieval_output_dir}/evidence_dict.pickle','wb') as file: # save results at each step
                pickle.dump(self.evidence_dict,file)

            # set back param priors for next retrieval
            if setback_prior==True:
                self.parameters.param_priors[f'log_{molecule}']=original_prior 
            
        return self.evidence_dict

    def compare_evidence(self,ln_Z_A,ln_Z_B):
        '''
        Convert log-evidences of two models to a sigma confidence level
        Originally from Benneke & Seager (2013)
        Adapted from samderegt/retrieval_base
        '''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = ln_Z_A-ln_Z_B
        sign=1
        if ln_B<0: # ln_Z_B larger -> second model favored
            sign=-1
            ln_B*=sign # can't handle negative values (-> nan), multiply back later
        p = np.real(np.exp(W((-1.0/(np.exp(ln_B)*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)
        return ln_B*sign,sigma*sign

    def run_retrieval(self,bayes=False): 

        print(f'\n ------ {self.target.name} - Nlive: {self.N_live_points} - ev: {self.evidence_tolerance} ------- \n')
        
        retrieval_output_dir=self.output_dir # save end results here
        molecules=['DMS','C2H6'] # list of molecules to run evidence retrievals on

        # run main retrieval if hasn't been run yet, else skip to evidence retrivals
        final_dict=pathlib.Path(f'{self.output_dir}/params_dict.pickle')
        if final_dict.exists()==False:
            print('\n ----------------- Starting main retrieval. ----------------- \n')
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
        else:
            print('\n ----------------- Main retrieval exists. ----------------- \n')
            with open(final_dict,'rb') as file:
                self.params_dict=pickle.load(file) 
        self.evaluate()
        if bayes==True:
            evidence_dict=pathlib.Path(f'{retrieval_output_dir}/evidence_dict.pickle')
            if evidence_dict.exists()==False: # to avoid overwriting sigmas from other evidence retrievals
                print('\n ----------------- Creating evidence dict ----------------- \n')
                self.evidence_dict={}
            else:
                print('\n ----------------- Continuing existing evidence dict ----------------- \n')
                with open(evidence_dict,'rb') as file:
                    self.evidence_dict=pickle.load(file)
            self.evidence_dict=self.bayes_evidence(molecules,retrieval_output_dir)
            with open(f'{retrieval_output_dir}/evidence_dict.pickle','wb') as file: # save new results in separate dict
                pickle.dump(self.evidence_dict,file)

        output_file=pathlib.Path('retrieval.out')
        if output_file.exists():
            os.system(f"mv {output_file} {retrieval_output_dir}")

        print('----------------- Done ----------------')

        
        

