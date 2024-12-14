test_dict={'log_g':(3.0,r'log $g$'),
            'log_H2O':(-2,r'log H$_2$O'),
            'log_CO':(-2,r'log CO'),
            'log_CO2':(-3,r'log CO$_2$'),
            'log_CH4':(-2,r'log CH$_4$'),
            'log_NH3':(-10,r'log NH$_3$'),
            'log_HCN':(-9,r'log HCN'),
            'log_H2S':(-7,r'log H$_2$S'),
            'log_C2H2':(-11,r'log C$_2$H$_2$'),
            'log_C2H4':(-6,r'log C$_2$H$_4$'),
            'log_C2H6':(-4,r'log C$_2$H$_6$'),
            'log_CH3Cl':(-7,r'log CH$_3$Cl'),
            'log_SO2':(-7,r'log SO$_2$'),
            'log_OCS':(-6,r'log OCS'),
            'log_CS2':(-8,r'log CS$_2$'),
            'log_DMS':(-7,r'log DMS'),
            'dlnT_dlnP_0': (0.3, r'$\nabla T_0$'), # gradient at T0 
            'dlnT_dlnP_1': (0.1, r'$\nabla T_1$'), 
            'dlnT_dlnP_2': (0.3, r'$\nabla T_2$'), 
            'dlnT_dlnP_3': (-0.01, r'$\nabla T_3$'), 
            'dlnT_dlnP_4': (0.2, r'$\nabla T_4$'), 
            'T0': (310, r'$T_0$')} # at bottom of atmosphere
            #'T0' : (350, r'$T_0$'), # bottom of the atmosphere (hotter)
            #'T1' : (260, r'$T_1$'),
            #'T2' : (200, r'$T_2$'),
            #'T3' : (220, r'$T_3$'),
            #'T4' : (210, r'$T_4$'),
            #'T5' : (220, r'$T_5$'),
            #'T6' : (200, r'$T_6$')} # top of atmosphere (cooler)

test_parameters={}
test_mathtext={}
for key_i, (value_i, mathtext_i) in test_dict.items():
   test_parameters[key_i] = value_i
   test_mathtext[key_i] = mathtext_i
test_PT= 'PTgrad'
test_chem='const'

# only execute code if run directly from terminal, otherwise just import params dict
if __name__ == "__main__":
      
    import config_run as cf
    from pRT_model import pRT_spectrum
    import matplotlib.pyplot as plt
    import numpy as np
    import pathlib
    import os 

    retrieval = cf.init_retrieval(obj='test',Nlive=100,evtol=1,PT_type=test_PT,chem=test_chem)
    retrieval.parameters.params = test_parameters
    data_wave= retrieval.data_wave
    test_object=pRT_spectrum(retrieval)
    test_spectrum=test_object.make_spectrum()
    test_spectrum/=np.nanmedian(test_spectrum) # normalize in same way as data spectrum
    white_noise=np.random.normal(np.zeros_like(test_spectrum),0.1,size=test_spectrum.shape)
    test_spectrum+=white_noise

    spectrum=np.full(shape=(len(test_spectrum),2),fill_value=np.nan)
    spectrum[:,0]=data_wave.flatten()
    spectrum[:,1]=test_spectrum

    output_dir = pathlib.Path(f'{os.getcwd()}/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(f'test/test_spectrum.txt',spectrum,delimiter=' ',header='wavelength(nm) flux')

    fig,ax=plt.subplots(1,1,figsize=(5,2),dpi=200)
    plt.plot(retrieval.data_wave,test_spectrum,lw=0.9,c=retrieval.target.color1)
    #fig.savefig(f'{output_dir}/test_spectrum.pdf',bbox_inches='tight')
    plt.show()

    fig,ax=plt.subplots(1,1,figsize=(3,3),dpi=200)
    plt.plot(test_object.temperature,test_object.pressure,c=retrieval.target.color1)
    plt.yscale('log')
    plt.gca().invert_yaxis()
    #fig.savefig(f'{output_dir}/test_PT.pdf',bbox_inches='tight')
