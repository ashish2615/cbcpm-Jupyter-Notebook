from __future__ import division, print_function

import os
import bilby
import numpy as np
import pandas as pd
import pickle
from time import process_time

from Initial_data import InitialData
from Injection import InjectionSignal

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory
outdir = './Subtractions'
if os.path.exists(outdir):
    print("Subtractions directory already exist")
else :
    print("Subtractions directory does not exist")
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

# bilby.utils.setup_logger(outdir=outdir, label='Subtractions')
        
class SubtractionSignal():

    def __init__(self):
        pass

    def subtraction_parameters(self, filename='./Injection_file/best_fit_params.pkl'):
        """
        Parameters
        -----------
        Return: Panda data frame
            Panda data frame of max likelihood parameter values 

        """
        
        file = open(filename,'rb')
        subtraction_param = pd.DataFrame(pickle.load(file))

        return subtraction_param

    def subtraction_signal(self, IFOs, sampling_frequency, seg_start_time, seg_end_time, t_coalescence, n_seg, N_samples, bestfit_params, waveform_generator):
        """
        Parameters
        ----------
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: float
            The GPS start-time of the data for n_seg.
        seg_end_time: float
            The GPS end-time of the data for n_seg.
        t_coalescence : float array
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment
        bestfit_params : dict
                Dictionary of max likelihood  of all best fit parameters
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Return:

        Calculating residual_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        sub_time_series: array like
                Array of time series of subtraction residuals.
        """
        
        print('Subtraction script starting (', process_time(), ')')
        
        subtracted = False
        tcnt = 0
        for index, single_params in bestfit_params.iterrows():
            
            #t_c = subtraction_parameters['geocent_time'][x]
            
            ## We used t_coalescence from the Injection.py script to have the same coalescence time for an Injected and Subtracted binary signal.
            t_c = t_coalescence[tcnt]
            
            single_params['geocent_time'] = t_c

            if seg_start_time < t_c < seg_end_time:

                subtracted = True

                print("Number of Subtraction signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                single_params['luminosity_distance'] = float(single_params['luminosity_distance'])

                ## First mass needs to be larger than second mass (just to cross check)
                if single_params['mass_1'] < single_params["mass_2"]:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp

                IFOs.subtract_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)
                tcnt += 1
                
        print('Signals subtracted (', process_time(), ')')
                
        #subtracted = False
        if subtracted:
            label = 'sub_segment_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            for ifo in IFOs:
                IFOs.plot_data(signal=None, outdir=outdir, label=label)
            print('Subtraction plots saved (', process_time(), ')')

        sub_time_series = np.zeros((len(IFOs), N_samples))
        cnt = 0
        for ifo in IFOs:
            sub_time_series[cnt, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            cnt += 1
            
        print('Residual time series calculated (', process_time(), ')')

        return sub_time_series, IFOs

