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
    #from LIFE_retrieval.pRT_model import pRT_spectrum
    #import LIFE_retrieval.figures as figs
    #from LIFE_retrieval.covariance import *
    #from LIFE_retrieval.log_likelihood import *
    from pRT_model import pRT_spectrum
    import figures as figs
    from covariance import *
    from log_likelihood import *
elif getpass.getuser() == "natalie": # when testing from my laptop
    os.environ['pRT_input_data_path'] = "/home/natalie/.local/lib/python3.8/site-packages/petitRADTRANS/input_data_std/input_data"
    from pRT_model import pRT_spectrum
    import figures as figs
    from covariance import *
    from log_likelihood import *


class Retrieval:

    def __init__(self,target,parameters,output_name,PT_type='PTgrad'):
        
        self.target=target
        self.data_wave,self.data_flux,self.data_err=target.load_spectrum()
        self.mask_isfinite=target.get_mask_isfinite() # mask nans
        self.parameters=parameters
        self.species=self.get_species(param_dict=self.parameters.params)

        self.n_pixels=len(self.data_wave)
        self.n_params = len(parameters.free_params)
        self.output_name=output_name
        self.cwd = os.getcwd()
        self.output_dir = pathlib.Path(f'{self.cwd}/{self.target.name}/{self.output_name}')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # cloud properties
        self.PT_type=PT_type
        self.lbl_opacity_sampling=1000
        self.n_atm_layers=50
        self.pressure = np.logspace(-6,2,self.n_atm_layers)

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
        self.final_params=None
        self.calc_errors=False

        self.color1=target.color1

    def get_species(self,param_dict): # get pRT species name from parameters dict
        species_info = pd.read_csv(os.path.join('species_info.csv'), index_col=0)
        self.chem_species=[]
        for par in param_dict:
            if 'log_' in par: # get all species in params dict, they are in log, ignore other log values
                if par in ['log_g','log_P_base_gray','log_opa_base_gray']: # skip
                    pass
                else:
                    self.chem_species.append(par)
        species=[]
        for chemspec in self.chem_species:
            species.append(species_info.loc[chemspec[4:],'pRT_name'])
        return species

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

            atmosphere_object = Radtrans(line_species=self.species,
                                rayleigh_species = ['H2', 'He'],
                                continuum_opacities = ['H2-H2', 'H2-He'],
                                wlen_bords_micron=wlen_range, 
                                mode='c-k')
            
            atmosphere_object.setup_opa_structure(self.pressure)
            with open(file,'wb') as file:
                pickle.dump(atmosphere_object,file)
            return atmosphere_object

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(parameters=self.parameters.params,
                                     data_wave=self.data_wave,
                                     target=self.target,
                                     atmosphere_object=self.atmosphere_object,
                                     species=self.species,
                                     PT_type=self.PT_type)
        
        # pass exit through a self.thing attibute and not kwarg, or pmn will be confused
        if self.calc_errors==True: # only for calc errors on Fe/H, C/O, temperatures
            return
        
        self.model_flux=self.model_object.make_spectrum()
        self.Cov(self.parameters.params)
        if True: # debugging
            plt.plot(self.data_wave,self.data_flux,lw=0.8)
            plt.plot(self.data_wave,self.model_flux,alpha=0.7,lw=0.8)
        ln_L = self.LogLike(self.model_flux, self.Cov) # retrieve log-likelihood
        return ln_L

    def PMN_run(self,N_live_points=400,evidence_tolerance=0.5,resume=False):
        pymultinest.run(LogLikelihood=self.PMN_lnL,Prior=self.parameters,n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True, sampling_efficiency = 0.5,
                        n_live_points=N_live_points,resume=resume,
                        evidence_tolerance=evidence_tolerance, # default is 0.5, high number -> stops earlier
                        dump_callback=self.PMN_callback,n_iter_before_update=100)

    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        #self.callback_label='live_' # label for plots
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        #np.save(f'{self.output_dir}/{self.callback_label}bestfit_params.npy',self.bestfit_params)
        self.posterior = posterior[:,:-2] # remove last 2 columns
        #np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.final_params,self.final_spectrum=self.get_params_and_spectrum()
        figs.summary_plot(self)
     
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

    def get_params_and_spectrum(self,save=False): 
        
        # make dict of constant params + evaluated params + their errors
        self.final_params=self.parameters.constant_params.copy() # initialize dict with constant params
        medians,minus_err,plus_err=self.get_quantiles(self.posterior)

        for i,key in enumerate(self.parameters.param_keys):
            self.final_params[key]=medians[i] # add median of evaluated params (more robust)
            self.final_params[f'{key}_err']=(minus_err[i],plus_err[i]) # add errors of evaluated params
            self.final_params[f'{key}_bf']=self.bestfit_params[i] # bestfit params with highest lnL (can differ from median, not as robust)

        # create final spectrum
        self.final_object=pRT_spectrum(parameters=self.final_params,data_wave=self.data_wave,
                                       target=self.target,species=self.species,
                                       atmosphere_object=self.atmosphere_object,
                                       contribution=True,
                                       PT_type=self.PT_type)
        self.final_model=self.final_object.make_spectrum()
        self.get_errors() # for temperature, C/O and [C/H]

        # get scaling parameters phi and s2 of bestfit model through likelihood
        self.log_likelihood = self.LogLike(self.final_model, self.Cov)
        self.final_params['phi']=self.LogLike.phi
        self.final_params['s2']=self.LogLike.s2
        if self.callback_label=='final_':
            self.final_params['chi2']=self.LogLike.chi2_0_red # save reduced chi^2 of fiducial model
            self.final_params['lnZ']=self.lnZ # save lnZ of fiducial model

        phi=self.final_params['phi']
        self.final_spectrum=phi*self.final_model # scale model accordingly
        
        spectrum=np.full(shape=(self.n_pixels,2),fill_value=np.nan)
        spectrum[:,0]=self.data_wave
        spectrum[:,1]=self.final_spectrum

        if save==True:
            with open(f'{self.output_dir}/{self.callback_label}params_dict.pickle','wb') as file:
                pickle.dump(self.final_params,file)
            np.savetxt(f'{self.output_dir}/{self.callback_label}spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')
        
        return self.final_params,self.final_spectrum

    def get_errors(self): # can only be run after self.evaluate()

        bounds_array=[]
        for key in self.parameters.param_keys:
            bounds=self.parameters.param_priors[key]
            bounds_array.append(bounds)
        bounds_array=np.array(bounds_array)

        self.calc_errors=True
        CO_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
        CH_distribution=np.full(self.posterior.shape[0],fill_value=0.0)
        temperature_distribution=[] # for each of the n_atm_layers
        x=0
        for j,sample in enumerate(self.posterior):
            # sample value is final/real value, need it to be between 0 and 1 depending on prior, same as cube
            cube=(sample-bounds_array[:,0])/(bounds_array[:,1]-bounds_array[:,0])
            self.parameters(cube)
            self.PMN_lnL()
            CO_distribution[j]=self.model_object.CO
            CH_distribution[j]=self.model_object.FeH
            temperature_distribution.append(self.model_object.temperature)
            x+=1
            if getpass.getuser()=="natalie" and x>20: # when testing from my laptop, or it takes too long (22min)
                break
        self.CO_CH_dist=np.vstack([CO_distribution,CH_distribution]).T
        self.temp_dist=np.array(temperature_distribution) # shape (n_samples, n_atm_layers)
        self.calc_errors=False # set back to False when finished

        median,minus_err,plus_err=self.get_quantiles(CO_distribution,flat=True)
        self.final_params['C/O']=median
        self.final_params['C/O_err']=(minus_err,plus_err)

        median,minus_err,plus_err=self.get_quantiles(CH_distribution,flat=True)
        self.final_params['C/H']=median
        self.final_params['C/H_err']=(minus_err,plus_err)

    def evaluate(self,callback_label='final_',save=False,makefigs=True):
        self.callback_label=callback_label
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.final_params,self.final_spectrum=self.get_params_and_spectrum(save=save) # all params: constant + free + scaling phi + s2
        if makefigs:
            if callback_label=='final_':
                figs.make_all_plots(self,split_corner=True)
            else:
                figs.summary_plot(self)

    def bayes_evidence(self,molecules):

        bayes_dict={}
        self.output_dir=pathlib.Path(f'{self.output_dir}/evidence_retrievals') # store output in separate folder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(molecules, list)==False:
            molecules=[molecules] # if only one, make list so that it works in for loop

        for molecule in molecules: # exclude molecule from retrieval

            finish=pathlib.Path(f'{self.output_dir}/evidence_retrievals/final_wo{molecule}_posterior.npy')
            if finish.exists():
                print(f'Evidence retrieval for {molecule} already done')
                continue # check if already exists and continue if yes

            original_prior=self.parameters.param_priors[f'log_{molecule}']
            self.parameters.param_priors[f'log_{molecule}']=[-15,-14] # exclude from retrieval
            self.callback_label=f'live_wo{molecule}_'
            self.prefix=f'pmn_wo{molecule}_' 
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance,resume=True)
            self.callback_label=f'final_wo{molecule}_'
            self.evaluate(callback_label=self.callback_label) # gets self.lnZ_ex

            ex_model=pRT_spectrum(parameters=self.final_params,data_wave=self.data_wave,
                                        target=self.target,species=self.species,
                                        atmosphere_object=self.atmosphere_object,
                                        contribution=True,
                                        PT_type=self.PT_type).make_spectrum()      
            lnL = self.LogLike(ex_model, self.Cov) # call function to generate chi2
            chi2_ex = self.LogLike.chi2_0_red # reduced chi^2

            print(f'lnZ=',self.lnZ)
            print(f'lnZ_{molecule}=',self.lnZ_ex)
            lnB,sigma=self.compare_evidence(self.lnZ, self.lnZ_ex)
            print(f'lnBm_{molecule}=',lnB)
            print(f'sigma_{molecule}=',sigma)
            print(f'chi2_{molecule}=',chi2_ex)
            bayes_dict[f'lnBm_{molecule}']=lnB
            bayes_dict[f'sigma_{molecule}']=sigma
            bayes_dict[f'chi2_wo_{molecule}']=chi2_ex
            print('bayes_dict=',bayes_dict)  

            # set back param priors for next retrieval
            self.parameters.param_priors[f'log_{molecule}']=original_prior 
            
        return bayes_dict

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

    def run_retrieval(self,N_live_points=400,evidence_tolerance=0.5,bayes=False): 
        self.N_live_points=N_live_points
        self.evidence_tolerance=evidence_tolerance
        retrieval_output_dir=self.output_dir # save end results here
        molecules=[] # list of molecules to run evidence retrievals on

        # run main retrieval if hasn't been run yet, else skip to evidence retrivals
        final_dict=pathlib.Path(f'{self.output_dir}/final_params_dict.pickle')
        if final_dict.exists()==False:
            self.PMN_run(N_live_points=self.N_live_points,evidence_tolerance=self.evidence_tolerance)
            save=True
        else:
            save=False
            with open(final_dict,'rb') as file:
                self.final_params=pickle.load(file) 
        self.evaluate(save=save)
        if bayes==True:
            bayes_dict=self.bayes_evidence(molecules)
            print('bayes_dict=\n',bayes_dict)
            with open(f'{retrieval_output_dir}/evidence_dict.pickle','wb') as file: # save new results in separate dict
                pickle.dump(bayes_dict,file)

        print('----------------- Done ----------------')

        
        

