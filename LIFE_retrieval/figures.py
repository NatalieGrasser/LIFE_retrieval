import getpass
import os
if getpass.getuser() == "grasser": # when runnig from LEM
    os.environ['OMP_NUM_THREADS'] = '1' # important for MPI

import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from pRT_model import pRT_spectrum

def plot_spectrum(retrieval_object,fs=10,**kwargs):

    wave=retrieval_object.data_wave
    flux=retrieval_object.data_flux
    err=retrieval_object.data_err
    flux_m=retrieval_object.model_flux

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(2,1,figsize=(9.5,3),dpi=200,gridspec_kw={'height_ratios':[2,0.7]})

    lower=flux-err*retrieval_object.params_dict['s2']
    upper=flux+err*retrieval_object.params_dict['s2']
    ax[0].plot(wave,flux,lw=1.2,alpha=1,c='k',label='data')
    ax[0].fill_between(wave,lower,upper,color='k',alpha=0.1,label=f'1 $\sigma$')
    ax[0].plot(wave,flux_m,lw=1.2,alpha=0.8,c=retrieval_object.color1,label='model')
    
    ax[1].plot(wave,flux-flux_m,lw=1.2,c=retrieval_object.color1,label='residuals')
    lines = [Line2D([0], [0], color='k',linewidth=2,label='Data'),
            mpatches.Patch(color='k',alpha=0.1,label='1$\sigma$'),
            Line2D([0], [0], color=retrieval_object.color1, linewidth=2,label='Bestfit')]
    ax[0].legend(handles=lines,fontsize=fs) # to only have it once
    ax[1].plot([np.min(wave),np.max(wave)],[0,0],lw=1.2,alpha=1,c='k')
        
    ax[0].set_ylabel('Normalized Flux',fontsize=fs)
    ax[1].set_ylabel('Residuals',fontsize=fs)
    ax[0].set_xlim(np.min(wave),np.max(wave))
    ax[1].set_xlim(np.min(wave),np.max(wave))
    tick_spacing=10
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)
    ax[1].set_xlabel('Wavelength [$\mu$m]',fontsize=fs)
    ax[0].get_xaxis().set_ticks([])

    plt.subplots_adjust(wspace=0, hspace=0)
    if 'ax' not in kwargs:
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}bestfit_spectrum.pdf',
                    bbox_inches='tight')
        plt.close()

def plot_pt(retrieval_object,fs=12,**kwargs):

    if 'ax' in kwargs:
        ax=kwargs.get('ax')
    else:
        fig,ax=plt.subplots(1,1,figsize=(5,5),dpi=200)

    lines=[]
    # plot PT-profile + errors on retrieved temperatures
    def plot_temperature(retr_obj,ax,olabel): 
        if retr_obj.PT_type=='PTknot':
            ax.plot(retr_obj.model_object.temperature,
                retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
            medians=[]
            errs=[]
            log_P_knots=retr_obj.model_object.log_P_knots
            for key in ['T4','T3','T2','T1','T0']: # order T4,T3,T2,T1,T0 like log_P_knots
                medians.append(retr_obj.params_dict[key])
                errs.append(retr_obj.params_dict[f'{key}_err'])
            errs=np.array(errs)
            for x in [1,2,3]: # plot 1-3 sigma errors
                lower = CubicSpline(log_P_knots,medians+x*errs[:,0])(np.log10(retr_obj.pressure))
                upper = CubicSpline(log_P_knots,medians+x*errs[:,1])(np.log10(retr_obj.pressure))
                ax.fill_betweenx(retr_obj.pressure,lower,upper,color=retr_obj.color1,alpha=0.15)
            ax.scatter(medians,10**retr_obj.model_object.log_P_knots,color=retr_obj.color1)
            xmin=np.min(lower)-100
            xmax=np.max(upper)+100
            lines.append(Line2D([0],[0],marker='o',color=retrieval_object.color1,markerfacecolor=retrieval_object.color1,
                    linewidth=2,linestyle='-',label=olabel))

        if retr_obj.PT_type=='PTgrad':
            dlnT_dlnP_knots=[]
            derr=[]
            for i in range(5):
                key=f'dlnT_dlnP_{i}'
                dlnT_dlnP_knots.append(retr_obj.params_dict[key]) # gradient median values
                derr.append(retr_obj.params_dict[f'{key}_err']) # -/+ errors
            derr=np.array(derr) # gradient errors
            T0=retr_obj.params_dict['T0']
            err=retr_obj.params_dict['T0_err']
            temperature=retr_obj.model_object.make_pt(dlnT_dlnP_knots=dlnT_dlnP_knots,T_base=T0)
            ax.plot(temperature,retr_obj.model_object.pressure,color=retr_obj.color1,lw=2) 
            # get 1-2-3 sigma of temp_dist, has shape (samples, n_atm_layers)
            quantiles = np.array([np.percentile(retr_obj.temp_dist[:,i], [0.2,2.3,15.9,50.0,84.1,97.7,99.8], axis=-1) for i in range(retr_obj.temp_dist.shape[1])])
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,0],quantiles[:,-1],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,1],quantiles[:,-2],color=retr_obj.color1,alpha=0.15)
            ax.fill_betweenx(retr_obj.pressure,quantiles[:,2],quantiles[:,-3],color=retr_obj.color1,alpha=0.15)
            xmin=np.min((quantiles[:,0],quantiles[:,-1]))-100
            xmax=np.max((quantiles[:,0],quantiles[:,-1]))+100
            lines.append(Line2D([0], [0], color=retr_obj.color1,
                                linewidth=2,linestyle='-',label=olabel))
        return xmin,xmax

    xmin,xmax=plot_temperature(retrieval_object,ax,olabel='Retrieval')
    model_object=pRT_spectrum(retrieval_object,contribution=True)
    summed_contr=model_object.contr_em
    contribution_plot=summed_contr/np.max(summed_contr)*(xmax-xmin)+xmin
    ax.plot(contribution_plot,retrieval_object.model_object.pressure,linestyle='dashed',
            lw=1.5,alpha=0.8,color=retrieval_object.color1)
    lines.append(Line2D([0], [0], color=retrieval_object.color1, alpha=0.8,
                        linewidth=1.5, linestyle='--',label='Em. Contr.'))
        
    ax.set(xlabel='Temperature [K]', ylabel='Pressure [bar]',yscale='log',
        ylim=(np.nanmax(retrieval_object.model_object.pressure),
        np.nanmin(retrieval_object.model_object.pressure)),xlim=(xmin,xmax))
    
    ax.legend(handles=lines,fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.set_xlabel('Temperature [K]', fontsize=fs)
    ax.set_ylabel('Pressure [bar]', fontsize=fs)

    if 'ax' not in kwargs: # save as separate plot
        fig.tight_layout()
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}PT_profile.pdf')
        plt.close()

def cornerplot(retrieval_object,getfig=False,figsize=(20,20),fs=12,plot_label='',
            only_abundances=False,only_params=None,not_abundances=False,truevals=False):
    
    plot_posterior=retrieval_object.posterior # posterior that we plot here, might get clipped
    medians,_,_=retrieval_object.get_quantiles(retrieval_object.posterior)
    labels=list(retrieval_object.parameters.param_mathtext.values())
    indices=np.linspace(0,len(retrieval_object.parameters.params)-1,len(retrieval_object.parameters.params),dtype=int)
    
    if only_abundances==True: # plot only abundances
        plot_label='_abundances'
        indices=[]
        for key in retrieval_object.chem_species:
            idx=list(retrieval_object.parameters.params).index(key)
            indices.append(idx)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])
        medians=np.array([medians[i] for i in indices])

    if only_params is not None: # keys of specified parameters to plot
        indices=[]
        for key in only_params:
            idx=list(retrieval_object.parameters.params).index(key)
            indices.append(idx)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in indices]).T
        labels=np.array([labels[i] for i in indices])
        medians=np.array([medians[i] for i in indices])

    if not_abundances==True: # plot all except abundances
        plot_label='_rest'
        abund_indices=[]
        for key in retrieval_object.chem_species:
            idx=list(retrieval_object.parameters.params).index(key)
            abund_indices.append(idx)
        set_diff = np.setdiff1d(indices,abund_indices)
        plot_posterior=np.array([retrieval_object.posterior[:,i] for i in set_diff]).T
        labels=np.array([labels[i] for i in set_diff])
        medians=np.array([medians[i] for i in set_diff])
        indices=set_diff

    fig = plt.figure(figsize=figsize) # fix size to avoid memory issues
    fig = corner.corner(plot_posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fs},
                        label_kwargs={'fontsize':fs*0.85},
                        color=retrieval_object.color1,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        hist_kwargs={'density': False,
                                'fill': True,
                                'alpha': 0.5,
                                'edgecolor': 'k',
                                'linewidth': 1.0},
                        fig=fig)
    
    # split title to avoid overlap with plots
    titles = [axi.title.get_text() for axi in fig.axes]
    for i, title in enumerate(titles):
        if len(title) > 30: # change 30 to 1 if you want all titles to be split
            title_split = title.split('=')
            titles[i] = title_split[0] + '\n ' + title_split[1]
        fig.axes[i].title.set_text(titles[i])

    # add true values of test spectrum
    # plotting didn't work bc x-axis range so small, some didn't show up
    if truevals==True:
        from testspec import test_parameters,test_mathtext
        compare=np.full(len(labels),None) # =None for non-input values of test spectrum
        for key_i in test_parameters.keys():
            label_i=test_mathtext[key_i]
            value_i=test_parameters[key_i]
            if label_i in labels:
                #print(label_i)
                #print(np.where(labels==label_i))
                i=np.where(labels==label_i)[0][0]
                compare[i]=value_i # add only those values that are used in cornerplot, in correct order
        x=0
        for i in range(len(compare)):
            titles[x] = titles[x]+'\n'+f'{compare[i]}'
            fig.axes[x].title.set_text(titles[x])
            x+=len(labels)+1

    plt.subplots_adjust(wspace=0,hspace=0)

    if getfig==False:
        fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}cornerplot{plot_label}.pdf',
                    bbox_inches="tight",dpi=200)
        plt.close()
    else:
        ax = np.array(fig.axes)
        return fig, ax

def make_all_plots(retrieval_object,only_params=None,split_corner=True):
    plot_spectrum(retrieval_object)
    plot_pt(retrieval_object)
    summary_plot(retrieval_object)
    if split_corner: # split corner plot to avoid massive files
        cornerplot(retrieval_object,only_abundances=True)
        cornerplot(retrieval_object,not_abundances=True)
    else: # make cornerplot with all parameters, could be huge, avoid this
        cornerplot(retrieval_object,only_params=only_params)
    
def summary_plot(retrieval_object,fs=14):

    # plot 7 most abundant species
    only_params=[]
    abunds=[]
    species=retrieval_object.chem_species
    for spec in species:
        abunds.append(retrieval_object.params_dict[spec])
    abunds, species = zip(*sorted(zip(abunds, species)))
    only_params=species[-7:] # get largest 7
    #only_params=['log_H2O','log_CO','log_CO2','log_CH4','log_NH3','log_H2S','log_HCN']
    fig, ax = cornerplot(retrieval_object,getfig=True,only_params=only_params,figsize=(17,17),fs=fs)
    l, b, w, h = [0.4,0.84,0.57,0.15] # left, bottom, width, height
    ax_spec = fig.add_axes([l,b,w,h])
    ax_res = fig.add_axes([l,b-0.03,w,h-0.12])
    plot_spectrum(retrieval_object,ax=(ax_spec,ax_res),fs=fs)

    l, b, w, h = [0.7,0.49,0.27,0.27] # left, bottom, width, height
    ax_PT = fig.add_axes([l,b,w,h])
    plot_pt(retrieval_object,ax=ax_PT,fs=fs)
    fig.savefig(f'{retrieval_object.output_dir}/{retrieval_object.callback_label}summary.pdf',
                bbox_inches="tight",dpi=200)
    plt.close()
