from __future__ import division, print_function

import os
import sys
import bilby
import numpy as np
import scipy
from scipy import integrate
import scipy.integrate as integrate

from Initial_data import InitialData

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)
## Specify the output directory and the name of the simulation.
outdir = 'ORF_OF'
label = 'ORF_OF'
# bilby.utils.setup_logger(outdir=outdir, label=label)

if os.path.exists('./ORF_OF'):
    print("ORF_OF directory already exist")
else :
    print("ORF_OF directory does not exist")
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

class detector_functions:

    def __init__(self):
        pass

    def initial(self, ifos, sampling_frequency, start_time, end_time, Tfft, n_seg):
        """
        Initialize/access the data from Initial_data script.
        check cbcpm.InitialData.initial_data

        ifos: iterable
            The list of interferometers
        sampling_frequency: float
            The sampling frequency (in Hz).
        duration_seg: float
            The data duration of a segment (in s).
        N_samples: int
            Number of samples in each segment of duration_duration_seg.
        frequency: array_like
           Real FFT of sample frequencies.

        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.

        """

        data = InitialData()
        data_sets = data.initial_data(ifos, sampling_frequency, start_time, end_time, Tfft, n_seg)
        self.sampling_frequency = data_sets[1]
        self.start_time = data_sets[2]
        self.end_time = data_sets[3]
        self.duration = data_sets[4]

        self.duration_seg = data_sets[5]
        self.n_seg = data_sets[6]
        self.N_samples =data_sets[7]
        self.frequency =data_sets[8]
        self.waveform_generator = data_sets[9]
        self.IFOs = data_sets[10]

        self.modes = data_sets[11]
        self.G = data_sets[12]
        self.one_pc = data_sets[13]
        self.H0 = data_sets[14]
        self.speed_of_light = data_sets[15]
        self.rho_c = data_sets[16]
        self.omega_gw = data_sets[17]
        
    def psd(self):
        '''
        Set One sided Noise Power Spectral Density of the detectors (because frequecy is from 0 to higher values)

        '''
        psd = np.zeros((len(self.IFOs), len(self.frequency)))
        ci = 0
        for ifo in self.IFOs:
            PSD = ifo.power_spectral_density
            psd[ci,] = PSD.power_spectral_density_interpolated(self.frequency)
            ci += 1
        psd[np.isinf(psd)] = 0
        psd[np.isnan(psd)] = 0

        return psd

    def overlap_reduction_function(self):
        """
        Return:
            Overlap Reduction function for a detector pair.
        """

        """" Calculate the position of the detector vertex in geocentric coordinates in meters. """
        position = np.zeros((len(self.IFOs), 3))
        cp = 0
        for ifo in self.IFOs:
            position[cp, :] = ifo.vertex_position_geocentric()
            cp += 1

        """Antena Response/Pattern factor for a given detector at given time and angles"""
        '''
        phi = ra:  right ascension in radians
        theta = dec: declination in radians
        '''
        ra_vec = np.linspace(0, 2 * np.pi, 100)
        dec_vec = np.linspace(-np.pi / 2, np.pi / 2, 100)

        ## d_omega = sin(theta) * d_theta * d_phi.
        d_dec = dec_vec[1] - dec_vec[0]
        d_ra = ra_vec[1] - ra_vec[0]
        d_sin = np.sin(dec_vec[1]) - np.sin(dec_vec[0])
        [ra_mat, dec_mat] = np.meshgrid(ra_vec, dec_vec)
        ra_vec = ra_mat.flatten()
        dec_vec = dec_mat.flatten()

        n_det = len(self.IFOs)
        
        antenna_response = np.zeros((n_det, len(self.modes), len(ra_vec)))
        ci = 0
        for ifo in self.IFOs:
            cm = 0
            for mode in self.modes:
                pidx = 0
                for pidx in range(len(ra_vec)):
                    antenna_response[ci, cm, pidx] = ifo.antenna_response(ra_vec[pidx], dec_vec[pidx], 0, 0, mode)
                    pidx += 1
                cm += 1
            ci += 1
            
        print('Antenna responses calculated.')

        
        ## orf = normalization calculated from antenna response function and overlap reduction function
        #orf = np.zeros((n_det,n_det, len(self.frequency)))
        ## cnst_orf = constant normalization factor 5/(8*np.pi) and overlap reduction function
        cnst_orf = np.zeros((n_det,n_det, len(self.frequency)))
        ## wn_orf = without normalization overlap reduction function
        #wn_orf = np.zeros((n_det,n_det, len(self.frequency)))
                           
        eins = np.ones(len(self.frequency))

        ## Iterate Over first detectors
        for d1 in range(n_det):
            f1p = antenna_response[d1, 0, :]  # mode = 0 i.e. plus polarisation
            f1c = antenna_response[d1, 1, :]  # mode = 1 i.e. cross polarisation
            ## Iterate Over second detectors
            for d2 in range(d1+1, n_det):
                f2p = antenna_response[d2, 0, :]
                f2c = antenna_response[d2, 1, :]

                delta_x = position[d1, :] - position[d2, :]
                omega = np.array([np.cos(dec_vec) * np.cos(ra_vec), np.cos(dec_vec) * np.sin(ra_vec), np.sin(dec_vec)])
                ## The unit vector Ω is a direction on the 2-D sphere (sky), described by two angles (θ, ϕ),
                ## from which the GW is arriving. We can therefore write the wavevector as k = 2πf Ω/c

                ## Normalization Constant, Correct
                #norm_orf1 = np.sum(np.cos(dec_vec) * (f1p * f1p + f1c* f1c)) * d_dec * d_ra
                #norm_orf2 = np.sum(np.cos(dec_vec) * (f2p * f2p + f2c* f2c)) * d_dec * d_ra
                #norm_orf12 = np.sqrt(norm_orf1*norm_orf2)
                
                ## Calculating overlap reduction function by using a normalization constant which depends upon the antenna response fucntion.
                #orf[d1, d2, :] = (norm_orf12)**(-1)  * np.sum( np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
                #        1j * 2 * np.pi * np.outer(self.frequency, np.dot(omega.T, delta_x)) / self.speed_of_light),axis=1) * d_dec * d_ra

                ## Calculating overlap reduction function by using a normalization constant (5/(8*np.pi))
                cnst_orf[d1, d2, :] = (5/(8*np.pi)) * np.sum(np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
                        1j * 2 * np.pi * np.outer(self.frequency, np.dot(omega.T, delta_x)) / self.speed_of_light),axis=1) * d_dec * d_ra

                ## Calculating overlap reduction function without using a normalization constant.
                #wn_orf[d1, d2, :] = np.sum(np.outer(eins, np.cos(dec_vec) * (f1p * f2p + f1c * f2c)) * np.exp(
                #        1j * 2 * np.pi * np.outer(self.frequency, np.dot(omega.T, delta_x)) / self.speed_of_light),axis=1) * d_dec * d_ra
                
                print ('Overlap-reduction function between', self.IFOs[d1].name, 'and', self.IFOs[d2].name, 'calculated.')

        return cnst_orf

    def plot_orf(self, gamma):

        # gamma = self.overlap_reduction_function()

        font = {'family': 'serif','color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size =6)

        n_det = len(self.IFOs)
        
        plt.subplot(2, 1, 1)
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, gamma[d1, d2, :], label=labelstring)

        legend = plt.legend(loc='lower right', prop=font1)
        plt.xlim(1, 1000)
        plt.xscale('log')
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.xlabel(r'f [Hz]', fontdict=font)
        plt.ylabel(r'$\gamma (f)$', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        #plt.title(r'ORF Between CE and ET_D_TR', fontdict=font)
        # plt.savefig('./ORF_OF/Overlap Reduction Function_log')
        # plt.show()
        # plt.close()

        plt.subplot(2, 1, 2)
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, gamma[d1, d2, :], label=labelstring)

        legend = plt.legend(loc='lower right', prop=font1)
        plt.xlim(0, 400)
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.axhline(y=0)
        # plt.axvline(x=0)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\gamma(f)$',fontdict=font)
        plt.tick_params(axis='both', direction='in')
        # plt.title(r'ORF Between CE and ET_D_TR')
        plt.tight_layout()
        plt.savefig('./ORF_OF/ORF_const_norm', dpi=300)
        plt.close()

    def optimal_filter_AR(self):
        '''
        Optimal filter depends upon the location and orientation of detector as well as SGWB and Noise PSD of detector.
        '''
        ## Allen and Romano 1999 Eq. 3.73
        ## wn_gamma = without normalization overlap reduction function.
        gamma, wn_gamma = overlap_reduction_function(self)
        psd = self.psd()
        
        n_det = len(self.IFOs)
        optimal_filter_AR = np.zeros((n_det, n_det, len(self.frequency)))
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                optimal_filter_AR[d1, d2, :] = (gamma[d1, d2, :] * self.omega_gw) / (self.frequency**3 * psd[d1, :] * psd[d2, :])

        return optimal_filter_AR

    def optimal_filter_JH(self, gamma):
        '''
        Optimal filter depends upon the location and orientation of detector as well as SGWB and Noise PSD of detector.
        '''

        psd = self.psd()

        ## Jan Paper
        ## Background spectral density S^b(frequency)
        Sb = np.dot((3. * self.H0**2) / (10 * np.pi**2), np.divide(self.omega_gw, self.frequency**3))
        # Sb = (3. * self.H0 ** 2) / (4 * np.pi ** 2) * self.omega_gw

        n_det = len(self.IFOs)
        
        optimal_filter_JH = np.zeros((n_det, n_det, len(self.frequency)))
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                optimal_filter_JH[d1, d2, :] = ((gamma[d1, d2, :] * Sb) / (psd[d1, :] * psd[d2, :]))

                optimal_filter_JH[np.isinf(optimal_filter_JH)] = 0
                optimal_filter_JH[np.isnan(optimal_filter_JH)] = 0

                ## Normalizing optimal filter between a range -1 to +1. using normalization equation
                norm_OF = optimal_filter_JH[d1, d2, :] / np.ptp(optimal_filter_JH[d1, d2, :])

                optimal_filter_JH[d1, d2,:] = norm_OF

        # optimal_filter_JH[np.isinf(optimal_filter_JH)] = 0
        # optimal_filter_JH[np.isnan(optimal_filter_JH)] = 0

        return optimal_filter_JH

    def plot_optfilter(self, optimal_filter):

        # optimal_filter = self.optimal_filter_JH()
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=6)

        n_det = len(self.IFOs)
        
        plt.subplot(2, 1, 1)
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, optimal_filter[d1, d2, :], label=labelstring)

        legend = plt.legend(loc='lower right', prop=font1)
        plt.xscale('log')
        plt.xlim(1, 1000)
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.xlabel(r'f (Hz)',fontdict=font)
        plt.ylabel(r'Q(f)',fontdict=font)
        plt.tick_params(axis='both', direction='in')
        #plt.title(r' Optimal filter Between CE and ET_D_TR', fontdict=font)
        # plt.savefig('./ORF_OF/ Optimal filter JH Between CE and ET_D_TR Function_log')
        # plt.show()
        # plt.close()

        plt.subplot(2, 1, 2)
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = self.IFOs[d1].name + ' & ' + self.IFOs[d2].name
                plt.plot(self.frequency, optimal_filter[d1, d2, :], label=labelstring)
        legend = plt.legend(loc='lower right', prop=font1)
        plt.xlim(0, 100)
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.axhline(y=0)
        # plt.axvline(x=0)
        plt.xlabel(r'Frequency $~\rm[Hz]$',fontdict=font)
        plt.ylabel(r'Q(f)',fontdict=font)
        plt.tick_params(axis='both', direction='in')
        # plt.title(r'$Optimal Filter Between CE and ET_D_TR$')
        plt.tight_layout()
        plt.savefig('./ORF_OF/Optimal_Filter_const_norm',dpi =300)
        plt.close()