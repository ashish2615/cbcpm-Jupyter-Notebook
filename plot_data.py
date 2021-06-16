from __future__ import division, print_function

import os
import sys
import bilby
import deepdish
import numpy as np
import logging
import deepdish
import pandas as pd
import json
import math
import sklearn
import seaborn as sns


from bilby.core.utils import speed_of_light

import scipy
from scipy import signal, fftpack
from scipy.fftpack import fft, rfft,ifft,irfft, fftfreq, rfftfreq
from scipy.signal import (periodogram, welch, lombscargle, csd, coherence,
                          spectrogram)
from scipy.signal import welch
from scipy.signal import *

import matplotlib
#matplotlib.use('tkagg')
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.font_manager as font_manager

# from Initial_data import InitialData, InterferometerStrain
# from Injection import InjectionSignal
# from Subtraction import SubtractionSignal
# from Projection import ProjectionSignal
# from ORF_OF import detector_functions
# from Cross_Correlation import CrossCorrelation

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)
## Specify the output directory and the name of the simulation.
outdir = 'Plot_PSD'
label = 'Plot_PSD'
# bilby.utils.setup_logger(outdir=outdir, label=label)

if os.path.exists('./Plot_PSD'):
    print("Plot_PSD directory already exist")
else :
    print("Plot_PSD directory does not exist")
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

class PSDWelch():

    def __init__(self):
        pass

    def plot_detector_psd(self, psd=None, frequency=None):
        """

        :param IFOs:
        :param psd:
        :param frequency:
        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        plt.loglog(frequency, np.sqrt(psd[0, :]), ':', label='aLIGO')
        # plt.loglog(frequency, np.sqrt(psd[1, :]), label='H1')
        plt.loglog(frequency, np.sqrt(psd[2, :]), '--', label='AdV')
        plt.loglog(frequency, np.sqrt(psd[3, :]), '-.', label='kAGRA')
        plt.loglog(frequency, np.sqrt(psd[4, :]), label='ET_D_TR')
        # plt.loglog(frequency, np.sqrt(psd[5, :]), label='ET_D_TR_2')
        # plt.loglog(frequency, np.sqrt(psd[6, :]), label='ET_D_TR_3')
        plt.loglog(frequency, np.sqrt(psd[7, :]), label='CE')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.xscale('log')
        plt.xlim(2, 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_Curve', dpi=300)
        plt.close()

    def plot_detector_psd_from_file(self, aLIGO_psd=None, AdV_psd=None, KAGRA_psd=None, aplus_LIGO_psd=None,
                                    aplus_V_psd=None, ET_D_TR_psd=None, CE_psd=None):
        """

        :param IFOs:
        :param psd:
        :param frequency:
        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')
        # plt.figure(figsize=(10, 8))

        plt.loglog(aLIGO_psd[:, 0], np.sqrt(aLIGO_psd[:, 1]), ':', label='aLIGO')
        plt.loglog(AdV_psd[:, 0], np.sqrt(AdV_psd[:, 1]), '--', label='AdV')
        plt.loglog(KAGRA_psd[:, 0], (KAGRA_psd[:, 1]), '-.', label='kAGRA')
        plt.loglog(aplus_LIGO_psd[:, 0], (aplus_LIGO_psd[:, 1]), label='$A^{+}$ LIGO')
        plt.loglog(aplus_V_psd[:, 0], (aplus_V_psd[:, 1]), label='$A^{+}$ VIRGO')
        plt.loglog(ET_D_TR_psd[:, 0], np.sqrt(ET_D_TR_psd[:, 1]), label='ET_D_TR')
        plt.loglog(CE_psd[:, 0], (CE_psd[:, 1]), label='CE')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        plt.xlim(1, 2 * 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain  [1/$\sqrt{\rm Hz}$] ', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve', dpi=300)
        plt.close()

    def plot_detector_psd_bilby_file(self, aLIGO_early_high=None, aLIGO_early=None, aLIGO_mid=None, aLIGO_late=None,
                                     aLIGO=None, aVIRGO=None, kagra=None, aplus=None, ligo_srd=None, ET=None, CE=None, CE1=None,CE2=None):
        """

        :param IFOs:
        :param psd:
        :param frequency:
        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        # plt.loglog(aLIGO_early_high[:,0], np.sqrt(aLIGO_early_high[:,1]), ':', label='aLIGO_early_high')
        # plt.loglog(aLIGO_early[:, 0], np.sqrt(aLIGO_early[:, 1]), '--', label='aLIGO_early')
        # plt.loglog(aLIGO_mid[:, 0], np.sqrt(aLIGO_mid[:, 1]), '-.', label='aLIGO_mid')
        # plt.loglog(aLIGO_late[:, 0], np.sqrt(aLIGO_late[:, 1]), label='aLIGO_late')
        plt.loglog(aLIGO[:, 0], np.sqrt(aLIGO[:, 1]), ':', label='aLIGO')
        plt.loglog(aVIRGO[:, 0], np.sqrt(aVIRGO[:, 1]), '--', label='aVIRGO')
        plt.    loglog(kagra[:, 0], np.sqrt(kagra[:, 1]),'-.', label='kAGRA')
        plt.loglog(aplus[:, 0], (aplus[:, 1]), label='$A^{+}$')
        # plt.loglog(ligo_srd[:, 0], np.sqrt(ligo_srd[:, 1]), label='ligo_srd')
        plt.loglog(ET[:, 0], np.sqrt(ET[:, 1]), label='ET')
        plt.loglog(CE[:, 0], np.sqrt(CE[:, 1]), label='CE')

        # plt.loglog(CE1[:, 0], (CE1[:, 1]), label='CE1')
        # plt.loglog(CE2[:, 0], (CE2[:, 1]), label='CE2')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.xscale('log')
        plt.xlim(1, 2 * 1000)
        plt.ylim(10**-25, 10**-18)
        plt.xlabel(r' Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r' Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for GW Detectors Bilby', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve_Bilby', dpi=300)
        plt.close()

    def plot_detector_spaced_based(self, LISA_psd=None, TianQin_psd=None, DECIGO_psd=None, BBO_psd=None, BBO1_psd=None):

        """
        :param LISA_psd:
        :param TianQin_psd:
        :param DECIGO_psd:
        :param BBO_psd:
        :param BBO1_psd:
        :return:
        """
        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='small')

        plt.loglog(LISA_psd[0], np.sqrt(LISA_psd[1]), label='LISA')
        plt.loglog(TianQin_psd[0], np.sqrt(TianQin_psd[1]), label='TianQin')
        plt.loglog(DECIGO_psd[0], np.sqrt(DECIGO_psd[1]), label='DECIGO')
        plt.loglog(BBO_psd[:, 0], (BBO_psd[:, 1]), label='BBO')
        plt.loglog(BBO1_psd[0], np.sqrt(BBO1_psd[1]), label='BBO1')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(10 ** -5, 6 * 10**2)
        plt.ylim(10 ** -25, 10 ** -14)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
        # plt.title(r'Sensitivity Curve for GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve_Spaced', dpi=300)
        plt.close()

    def plot_detector_psd_All(self, aLIGO_psd=None, AdV_psd=None, KAGRA_psd=None, aplus_LIGO_psd=None, aplus_V_psd=None,
                              ET_D_TR_psd=None, CE_psd=None, BBO_psd=None, LISA_psd=None, DECIGO_psd=None,
                              BBO1_psd=None, TianQin_psd=None,PGWB=None, cosmo_spectrum=None):
        """

        :param IFOs:
        :param psd:
        :param frequency:
        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 20}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=12)
        # plt.rcParams["figure.figsize"] = (13,6)
        plt.figure(figsize=(12, 8))

        plt.loglog(aLIGO_psd[:, 0], np.sqrt(aLIGO_psd[:, 1]), ':', label='aLIGO')
        plt.loglog(AdV_psd[:, 0], np.sqrt(AdV_psd[:, 1]), '--', label='AdV')
        plt.loglog(KAGRA_psd[:, 0], (KAGRA_psd[:, 1]), '-.', label='kAGRA')
        plt.loglog(aplus_LIGO_psd[:, 0], (aplus_LIGO_psd[:, 1]), label='$A^{+}$ LIGO')
        plt.loglog(aplus_V_psd[:, 0], (aplus_V_psd[:, 1]), label='$A^{+}$ VIRGO')
        plt.loglog(ET_D_TR_psd[:, 0], np.sqrt(ET_D_TR_psd[:, 1]), label='ET_D_TR')
        plt.loglog(CE_psd[:, 0], (CE_psd[:, 1]), label='CE')

        plt.loglog(BBO_psd[:, 0], (BBO_psd[:, 1]), label='BBO')
        plt.loglog(DECIGO_psd[0], np.sqrt(DECIGO_psd[1]), label='DECIGO')
        plt.loglog(BBO1_psd[0], np.sqrt(BBO1_psd[1]), label='BBO1')

        plt.loglog(LISA_psd[0], np.sqrt(LISA_psd[1]), label='LISA')
        plt.loglog(TianQin_psd[0], np.sqrt(TianQin_psd[1]), label='TianQin')

        plt.loglog(PGWB[0], PGWB[1], '--' , label='Primordial GWB')
        plt.text(0.5, 1*10**-26, 'Primordial GWB', horizontalalignment='right',color='green',fontsize=16, weight='bold')

        # frequency = np.logspace(-18, 4, num=8193)
        # plt.loglog(frequency, cosmo_spectrum, label='$\Omega_{GW}^{cosmo}$')

        legend = plt.legend(loc='upper right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.xlim(10 ** -4,  10000)
        plt.ylim(10 ** -26, 10 ** -14)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)

        # plt.rcParams["figure.figsize"] = (20,20)
        # plt.figure(figsize=(1, 1))
        # plt.tick_params(axis='both', direction='in')
        # plt.title(r'Sensitivity Curve for Space and Ground based GW Detectors', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Sensitivity_PSD_Curve_All', dpi=300)
        plt.close()

    def plot_strain2omega(self,aLIGO=None, AdV=None, KAGRA=None, aplus_LIGO=None, aplus_V=None, ET_D_TR=None, CE=None,
                          BBO=None, LISA=None, DECIGO=None, BBO1=None, TianQin=None, cosmo_spectrum=None):
        """

        Check Sensitivity_curve  SensitivityCurve.strain2omega()
        :param aLIGO:
        :param AdV:
        :param KAGRA:
        :param aplus_LIGO:
        :param aplus_V:
        :param ET_D_TR:
        :param CE:
        :param BBO:
        :param LISA:
        :param DECIGO:
        :param BBO1:
        :param TianQin:
        :return:
        """

        font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 14}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size=10)
        # plt.rcParams["figure.figsize"] = (13,6)
        # plt.figure(figsize=(12, 8))

        # plt.loglog(aLIGO[0], (aLIGO[1]), ':', label='aLIGO')
        # plt.loglog(AdV[0], (AdV[1]), '--', label='AdV')
        # plt.loglog(KAGRA[0], (KAGRA[1]), '-.', label='kAGRA')
        # plt.loglog(aplus_LIGO[0], (aplus_LIGO[1]), label='$A^{+}$ LIGO')
        # plt.loglog(aplus_V[0], (aplus_V[1]), label='$A^{+}$ VIRGO')
        # plt.loglog(ET_D_TR[0], (ET_D_TR[1]), label='ET_D_TR')
        # plt.loglog(CE[0], (CE[1]), label='CE')

        plt.loglog(BBO[0], (BBO[1]), label='BBO')
        plt.loglog(DECIGO[0], (DECIGO[1]), label='DECIGO')
        plt.loglog(BBO1[0], (BBO1[1]), label='BBO1')

        plt.loglog(LISA[0], (LISA[1]), label='LISA')
        plt.loglog(TianQin[0], (TianQin[1]), label='TianQin')
        #
        # frequency = np.logspace(-18, 4, num=8193)
        # plt.loglog(frequency, cosmo_spectrum, label='$\Omega_{GW}^{cosmo}$')

        legend = plt.legend(loc='lower right', prop=font1)
        # plt.grid(True, which="majorminor", ls="-", color='0.5')
        # plt.xscale('log')
        # plt.yscale('log')

        ##Ground Based Limit
        # plt.xlim(1, 2 * 1000)
        # plt.ylim(10 ** -14, 10 ** -1)
        ## Space based limit
        plt.xlim(10**-4, 100)
        plt.ylim(10 ** -20, 10 ** -1)
        ## X and Y limit for All
        # plt.xlim(10**-5, 4 * 1000)
        # plt.ylim(10 ** -20, 10 ** -1)

        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\Omega_{GW}(f)$ ', fontdict=font)
        plt.tick_params(axis='both', direction='in')

        # plt.rcParams["figure.figsize"] = (20,20)
        # plt.figure(figsize=(1, 1))
        plt.tick_params(axis='both', direction='in')
        # plt.title(r'Strain to Omega', fontdict=font)
        plt.tight_layout()
        plt.savefig('./Sensitivity_Curve/Strain_to_Omega_Space', dpi=300)
        plt.close()

    def plot_data(self, IFOs, sampling_frequency, n_seg, inj_time_series, sub_time_series, proj_time_series, Tfft):
        """
        IFOs: Initialization of GW interferometer.
           Generates an Interferometer with a power spectral density.
        sampling_frequency: float
            The sampling frequency (in Hz).
        n_seg: int
            number of time segment in total time duration of operation of detector.
        inj_time_series : array_like
            time_domain_strain for injection signals in units of strain.
        sub_time_series: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        proj_data_stream: array like
            Real valued array of time series calculated by taking inverse fft of residual_noise_data.
        Tfft: float
            FFT time.
            
        Note: check scipy.signal.welch for to use default values.

        Return:

        *** = inj, sub, proj

        freq_*** : ndarray
            Array of sample frequencies.
        ***_welch : ndarray
            Power spectral density or power spectrum of ***_time_series.

        """

        for detector in range(len(IFOs)):

            font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            nperseg = int(sampling_frequency * Tfft)

            ifo = IFOs[detector].name
            label = ifo + ' After Injections'
            label1 = ifo + ' After Subtraction'
            label2 = ifo + ' After Projection'

            inj_freq, inj_welch = scipy.signal.welch(inj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('inj_welch',inj_welch)
            sub_freq, sub_welch = scipy.signal.welch(sub_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('sub_welch',sub_welch)
            proj_freq, proj_welch =  scipy.signal.welch(proj_time_series[detector, :], fs = sampling_frequency, nperseg=nperseg)
            # print('proj_welch',proj_welch)

            plt.subplot(3, 1, 1)
            plt.semilogy(inj_freq, np.sqrt(inj_welch), label=label)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'f (Hz)')
            # plt.ylabel(r'PSD_Welch(f)')
            #plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)

            plt.subplot(3, 1, 2)
            plt.semilogy(sub_freq, np.sqrt(sub_welch), label=label1)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            # plt.xlabel(r'f (Hz)')
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)

            plt.subplot(3, 1, 3)
            plt.semilogy(proj_freq, np.sqrt(proj_welch), label=label2)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xscale('log')
            plt.xlim(10, 1000)
            # plt.ylim(10**-20, 10**-32)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$',fontdict=font)
            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.5)
            # plt.ylabel(r'PSD_Welch(f)')
            plt.savefig('./Plot_PSD/PSD_Welch_'+ifo+'_'+str(n_seg)+'.png', dpi=300)
            # plt.savefig('./Plot_PSD/PSD_Welch_{}_{}_{}_{}'.format(ifo, str(n_seg), detector, nperseg)) #+ifo+str(int(n_seg))+'_'+str(detector)+str(nperseg)
            plt.close()

            plt.loglog(inj_freq, np.sqrt(inj_welch), 'r-',  label=label)
            plt.loglog(sub_freq, np.sqrt(sub_welch), 'b-', label=label1)
            plt.loglog(proj_freq, np.sqrt(proj_welch), 'g-', label=label2)
            legend = plt.legend(loc='best', prop=font1)
            # plt.xscale('log')
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -34)
            # plt.autoscale(enable=True, axis='y', tight=False)
            plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            #plt.title(r' PSD Spectrum for ' + ifo, fontdict=font)
            plt.savefig('./Plot_PSD/PSD_Welch_comparison_'+ifo+'_'+str(n_seg)+'.png',dpi=300)
            plt.close()

    def plot_one_psd(self, IFOs, sampling_frequency, n_seg, time_series, Tfft):
        """
        :param IFOs:
        :param sampling_frequency:
        :param n_seg:
        :param time_series:
        :param Tfft:
        :return:
        """

        nperseg = int(sampling_frequency * Tfft)

        for detector in range(len(IFOs)):
            ifo = IFOs[detector].name
            label = ifo + ' After Injections'
            label1 = ifo + ' After Subtraction'
            label2 = ifo + ' After Projection'

            freq, welch = scipy.signal.welch(time_series[detector, :], fs=sampling_frequency, nperseg=nperseg, nfft=nperseg)

            font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal','size': 12}
            font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

            plt.loglog(freq, np.sqrt(welch), label=label)
            legend = plt.legend(loc='lower left', prop=font1)
            plt.xlim(1, 1000)
            # plt.ylim(10 ** -20, 10 ** -32)
            plt.xlabel(r' Frequency $~\rm[Hz]$', fontdict=font)
            plt.ylabel(r'Strain [1/$~\sqrt{\rm Hz}$]', fontdict=font)
            plt.tight_layout()
            plt.savefig('./Plot_PSD/PSD_'+ifo+'_'+str(n_seg)+'.png', dpi=300)
            plt.close()


    def plots_csd(self,IFOs, frequency, psd_series=None, cc_inj=None, cc_sub=None, cc_proj=None, n_seg=None):
        """
        :param IFOs:
        :param frequency:
        :param psd_series:
        :param cc_inj:
        :param cc_sub:
        :param cc_proj:
        :param n_seg:
        :return:
        """

        n_det = len(IFOs)
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):

                font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
                font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

                # plt.loglog(frequency, psd_series[d1,d2,:], 'y-', label='Noise')
                plt.loglog(frequency, np.abs(cc_inj[d1,d2,:]), 'r-', label='injection')
                plt.loglog(frequency, np.abs(cc_sub[d1,d2,:]), 'b-', label='subtraction')
                plt.loglog(frequency, np.abs(cc_proj[d1,d2,:]), 'g-', label='projection')
                legend = plt.legend(loc='lower left', prop=font1)
                # plt.xscale('log')
                plt.xlim(1, 1000)
                # plt.ylim(10 ** -20, 10 ** -34)
                plt.autoscale(enable=True, axis='y', tight=False)
                plt.xlabel('Frequency [Hz]', fontdict=font)
                plt.ylabel('CSD  [1/Hz]', fontdict=font)
                plt.tight_layout()
                # plt.title(r' CSD Spectrum for_' + str(detector), fontdict=font)
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name
                plt.savefig('./Plot_PSD/CSD_comparision_'+str(n_seg)+ '_' + labelstring +'.png', dpi=300)
                plt.close()

    def plot_avg_csd(self, IFOs, frequency, sum_series=None, method=None, n_seg=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param method:
        :param n_seg:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        n_det = len(IFOs)
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):

                plt.loglog(frequency, (np.abs(sum_series[d1,d2, :])), label=method)
                legend = plt.legend(loc='lower left', prop= font1)
                plt.xlim(1, 1000)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r' CSD [1/$\sqrt{\rm Hz}$]', fontdict=font)
                plt.tight_layout()
                #plt.title(r'Avg Cross_Correlation for ' +method, fontdict=font)
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name
                plt.savefig('./Cross_Corr/Avg_Cross_Correlation_' + labelstring + '_' + method, dpi=300)
                plt.close()

    def plot_variance(self, IFOs, frequency, variance=None, n_seg=None, method=None):
        """
        :param frequency:
        :param variance:
        :return:
        """
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        n_det = len(IFOs)
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name

                plt.loglog(frequency, variance[d1,d2, :])
                plt.xlim(1, 1000)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'$ \sigma ~~ \rm[1/Hz]$',fontdict=font)
                plt.tight_layout()
                #plt.title(r'Variance Between_' + str(ifo) + '_' + method, fontdict=font)
                plt.savefig('./Cross_Corr/variance_noise_' + labelstring +'_' + str(n_seg) + '_' + method, dpi=300)
                plt.close()

    def plot_csd_var(self, IFOs, frequency, sum_series=None, variance=None, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None, n_seg=None, method=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param variance:
        :param n_seg:
        :param method:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        n_det = len(IFOs)
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name

                # for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):
                plt.loglog(frequency, np.abs(sum_series[d1,d2, :]), label='CSD')
                plt.loglog(frequency, variance[d1,d2, :], label='Sigma')
                plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1,d2,:]), label='CSD_Astro')
                plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1,d2, :]), label='CSD_Cosmo')

                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(1*10**-65, 1*10**-40)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD $~ & ~\sigma$ [1/Hz]', fontdict=font)
                plt.tight_layout()
                #plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
                plt.savefig('./Cross_Corr/CSD_Variance_Astro_Cosmo_'  + labelstring + '_' + method + '_' + str(n_seg), dpi=300)
                plt.close()

    def plot_sum_csd_var(self, IFOs, frequency, sum_series=None, variance=None, CSD_from_Omega_astro=None, CSD_from_Omega_cosmo=None, n_seg=None, method=None, outdir=None):
        """
        :param IFOs:
        :param frequency:
        :param sum_series:
        :param variance:
        :param n_seg:
        :param method:
        :return:
        """
        one_pc = 3.0856775814671916 * 10 ** 16  ## units = m
        H0 = 67.9 * 10 ** 3 * 10 ** -6 * one_pc ** -1  ## units = 1/sec

        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        n_det = len(IFOs)
        
        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name

                # for idx in np.arange((int(len(IFOs) * (len(IFOs) - 1) / 2))):
                plt.loglog(frequency, np.abs(sum_series[d1,d2, :]), label='CSD')
                plt.loglog(frequency, variance[d1,d2, :], label='Sigma')
                plt.loglog(frequency, np.abs(CSD_from_Omega_astro[d1,d2,:]), label='CSD_Astro')
                plt.loglog(frequency, np.abs(CSD_from_Omega_cosmo[d1,d2, :]), label='CSD_Cosmo')

                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                # plt.ylim(1*10**-60, 1*10**-35)
                # plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r'CSD $~ & ~\sigma ~~$ [1/Hz]', fontdict=font)
                plt.tight_layout()
                #plt.title(r'CSD_and_Variance for_' +str(ifo)+'_'+ method, fontdict=font)
                plt.savefig(outdir + '/CSD_and_Variance_and_CSD_Astro_'  + labelstring + '_' + method +'_' + str(n_seg), dpi=300)
                plt.close()

    def plot_CSD_from_Omega(self, IFOs, frequency, CSD_from_Omega=None, background=None):
        """
        :param IFOs:
        :param frequency:
        :param CSD_from_Omega:
        :return:
        """

        n_det = len(IFOs)
        
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        for d1 in range(n_det):
            for d2 in range(d1+1, n_det):
                labelstring = IFOs[d1].name + ' & ' + IFOs[d2].name

                plt.loglog(frequency, np.sqrt(np.abs(np.real(CSD_from_Omega[d1,d2,:]))), label='CSD_'+ background)
                legend = plt.legend(loc='lower left', prop=font1)
                plt.xlim(1, 1000)
                plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
                plt.ylabel(r' CSD [1/$\sqrt{\rm Hz}$]', fontdict=font)
                plt.tight_layout()
                #plt.title(r'CSD_from_Omega_' + str(ifo) + background, fontdict=font)
                plt.savefig('./Omega_gw/CSD_from_Omega_' + labelstring +'_'+background, dpi=300)
                plt.close()


    def plot_comso_omega_gw(self,frequency,comso_omega_gw=None, cobe_spectrum=None, PGWB=None):
        """
        :param comso_omega_gw: 
        :param frequency: 
        :return: 
        """
        font = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': 12}
        font1 = font_manager.FontProperties(family='serif', weight='normal', style='normal', size='xx-small')

        one_pc = 3.0856775814671916 * 10**16  ## units = m
        H00 = 67.9 * 10**3 * 10**-6 * one_pc**-1  ## units = 1/sec
        H0 = 30 * H00
        freq = np.logspace(-18, -16, num=len(frequency))
        frequency = np.logspace(-18, 4, num=len(frequency))

        plt.loglog(frequency, comso_omega_gw, label='$\Omega_{GW}^{cosmo}$')
        plt.loglog(freq, cobe_spectrum, label ='COBE')
        plt.loglog(PGWB[0], PGWB[1], '--', label='Primordial GWB')
        plt.text(0.8, 10 **-18, 'Primordial GWB', horizontalalignment='right', color='green', fontsize=12)

        legend = plt.legend(loc='lower left', prop=font1)
        plt.xlim(10**-18, 10**4)
        plt.ylim(10**-20, 10**-10)
        plt.xlabel(r'Frequency $~\rm[Hz]$', fontdict=font)
        plt.ylabel(r'$\Omega(f)$', fontdict=font)
        plt.tight_layout()
        #plt.title(r'$\Omega_{GW}^{cosmo}$', fontdict=font)
        plt.savefig('./Omega_gw/Cosmo_Omega_GW_COBE', dpi=300)
        plt.close()


