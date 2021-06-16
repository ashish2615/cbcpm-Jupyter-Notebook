from __future__ import division, print_function

import os
import bilby
import random
import numpy as np
import pandas as pd
from time import process_time

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)

## Specify the output directory
outdir = './Injections'
if os.path.exists(outdir):
    print("Injections directory already exist")
else :
    print("Injections directory does not exist")
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

# bilby.utils.setup_logger(outdir=outdir, label='Injections')

class InjectionSignal:

    def __init__(self):
        pass

    def injection_parameters(self, filename=None):
        """
        List of injection signals sorted by redshift.

        filename: str, Data file
            Injection signal Data file

        Return: data file, hdf5
                Injection signals in asecnding order in redshift.
        """
        injections = pd.read_hdf(filename)
        injections = injections.sort_values('redshift', ascending=True)
        injections.index = range(len(injections.index))

        return injections

    def injection_signal(self, IFOs, sampling_frequency, seg_start_time, seg_end_time,
                         n_seg, N_samples, injection_params, waveform_generator):
        """
        Injection signals : Signals which are known and identified inspiralling compact binaries from the data_stream.

        Parameters
        -------------
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        sampling_frequency: float
            The sampling frequency (in Hz).
        seg_start_time: 
            The GPS start-time of the data for n_seg.
        seg_end_time:
            The GPS end-time of the data for n_seg.
        n_seg: int
            number of time segment in total time duration of operation of detector
        N_samples: int
            Number of samples in each segment
        injection_params: dict
            Parameters of the injection.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Return:
        -----------
        Calculating time_doamin_strain for all detectors.
        Check bilby/gw/detector/strain_data.py for more details

        inj_time_series : array_like
                    time_domain_strain for injection signals in units of strain.
        """

        print('Injection script starting (', process_time(), ')')
        
        ## Changing 'iota' to 'theta_jn' to be suitable with bilby
        injection_params['theta_jn'] = injection_params['iota']
            
        injected = False
        
        ## Defining a t_coalescence array to store the coalescence time of a binary signal genrated randomly 
        ## between the seg_start_time and seg_end_time.
        t_coalescence = np.zeros(len(injection_params))
        
        tcnt = 0
        for index, single_params in injection_params.iterrows():
            
            ## Generating a random array of coalescence time between the seg_start_time and seg_end_time of the time_seg
            t_c = random.uniform(np.float(seg_start_time+10), np.float(seg_end_time-10))
            
            single_params['geocent_time'] = t_c
        
            if seg_start_time < t_c < seg_end_time:

                injected = True

                print("Number of Injection signal is :", index)
                print('seg_start_time', seg_start_time)
                print('geocent_time', t_c)
                print('seg_end_time', seg_end_time)

                ## Saving t_coalescence as an array for Subtraction part
                t_coalescence[tcnt] = t_c
                
                ## Redshift to luminosity Distance conversion using bilby
                single_params['luminosity_distance'] = bilby.gw.conversion.redshift_to_luminosity_distance(single_params['redshift'])

                ## First mass needs to be larger than second mass
                if single_params['mass_1'] < single_params['mass_2']:
                    tmp = single_params['mass_1']
                    single_params['mass_1'] = single_params['mass_2']
                    single_params['mass_2'] = tmp
                
                IFOs.inject_signal(parameters=single_params.to_dict(), waveform_generator=waveform_generator)
                tcnt += 1
        
        print('Signals injected (', process_time(), ')')
        
        #injected = False
        if injected:
            label = 'inj_segment_' + str(n_seg)
            # IFOs.save_data(outdir=outdir, label=label)
            for ifo in IFOs:
                IFOs.plot_data(signal=None, outdir=outdir, label=label)
            print('Injection plots saved (', process_time(), ')')

        inj_time_series = np.zeros((len(IFOs), (N_samples)))
        ci = 0
        for ifo in IFOs:
            inj_time_series[ci, :] = bilby.core.utils.infft(ifo.strain_data.frequency_domain_strain, sampling_frequency)
            ci += 1
            
        print('Time series calculated (', process_time(), ')')

        return t_coalescence, inj_time_series, IFOs


