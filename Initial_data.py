from __future__ import division, print_function

import os
import sys
import bilby
from bilby.core.utils import speed_of_light
import numpy as np
from numpy.core._multiarray_umath import dtype
import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft,ifft,irfft, fftfreq, rfftfreq

import astropy
from astropy import constants as const
from astropy.constants import G, pc, c

class InitialData:

    def __init__(self):
        pass

    def initial_data(self, ifos, sampling_frequency, start_time, end_time, Tfft, n_seg=None):

        """
        Parameteres
        -----------
        ifos: List of strings
            List of all detectors
        sampling_frequency: float
            The sampling frequency (in Hz).
        start_time: float
            start time (GPS) of simulated data
        end_time:  float
            end time (GPS) of simulated data
        Tfft: int
            length of time segment for FFTs
        n_seg:  int
            number of segments in total time duration of simulated data


        Return:
        -------------
        ifos: List
            List of all detectors.
        sampling_frequency: float
            The sampling frequency (in Hz).
        start_time: float
            data taking or operation of period detector. The GPS start-time of the data.
        end_time: float
            the end time of the total simulated data
        n_seg: int
            number of segment in total time duration of operation of detector.
        N_samples: int
            Number of samples in each segment of duration duration_seg.
        frequency: array_like
           Real FFT of sample frequencies.

        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.

        Gravitational Constant G: int
        one_pc: int
            distance, one persec in meters
        H0: int
            Hubble constant
        speed_of_light: int
            constant
        rho_c: int
            critical energy density required to close the Universe
        omega_gw:int
            Energy density of Stochastic Cosmological background Gravitational Wave. omega_gw is a dimensionless quantity.

        """

        self.ifos = ifos
        self.sampling_frequency = sampling_frequency
        self.start_time = start_time
        self.end_time = end_time

        if n_seg is None:
            n_seg = 10000
        else:
            n_seg = n_seg

        self.n_seg = n_seg

        self.duration = self.end_time - self.start_time
        duration_seg = self.duration / n_seg

        self.duration_seg = 2 ** (int(duration_seg) - 1).bit_length()

        n_seg = np.trunc(self.duration / self.duration_seg)

        self.N_samples = int(self.sampling_frequency * self.duration_seg)

        delta_T = self.duration_seg / self.N_samples
        frequency_resolution = 1 / self.duration_seg

        n_fft = Tfft * self.sampling_frequency

        n_frequencies = n_fft//2 + 1
        freq_series = np.linspace(start=0, stop = self.sampling_frequency / 2, num = n_frequencies)

        freq_rfft = rfftfreq(n_fft, d=1. / self.sampling_frequency)

        self.frequency = np.append([0], freq_rfft[1::2])

        self.waveform_arguments = dict(waveform_approximant='IMRPhenomPv2', reference_frequency=50.,minimum_frequency=2.)

        self.waveform_generator = bilby.gw.WaveformGenerator(duration=self.duration_seg, sampling_frequency=self.sampling_frequency,
                                                                   frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                                   waveform_arguments=self.waveform_arguments)

        self.IFOs = bilby.gw.detector.networks.InterferometerList(ifos)

        self.modes = ['plus', 'cross']  # ,'breathing']

        ## Hubble constant H0 = (67.4±0.5) km s−1Mpc−1
        self.G = 6.67408 * 10 ** -11  ## units = m**3/ (kg* sec**2)
        self.one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        self.H0 = 67.9 * 10 ** 3 * 10 ** -6 * self.one_pc ** -1  ## units = 1/sec
        # self.h0 = 0.6766
        # self.H0 = h0*3.24*10**-18
        self.speed_of_light = bilby.gw.utils.speed_of_light

        ## rho_c = (3 * c ** 2 * H0 ** 2)/(8 * np.pi * G)
        self.rho_c = (3 * self.speed_of_light**2 * self.H0**2) / (8 * np.pi * self.G)  ## units = erg/cm**3
        self.omega_gw = 10**-15

        data_set = (self.ifos, self.sampling_frequency, self.start_time, self.end_time, self.duration, self.duration_seg, self.n_seg,
                    self.N_samples, self.frequency, self.waveform_generator, self.IFOs, self.modes, self.G, self.one_pc,self.H0, self.speed_of_light,self.rho_c, self.omega_gw)

        return data_set
