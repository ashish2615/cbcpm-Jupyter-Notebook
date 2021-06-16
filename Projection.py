from __future__ import division, print_function

import os
import bilby
import numpy as np
from time import process_time

current_direc = os.getcwd()
# print("Current Working Directory is :", current_direc)ù

## Specify the output directory
outdir = './Projections'
if os.path.exists(outdir):
    print("Projections directory already exist")
else :
    print("Projections directory does not exist")
    try:
        os.mkdir(outdir)
    except OSError:
        print("Creation of the directory {} failed".format(outdir))
    else:
        print("Successfully created the directory {}".format(outdir))

# bilby.utils.setup_logger(outdir=outdir, label='Projections')
        
class ProjectionSignal:

    def __init__(self):
        pass

    def projection_derivatives(self, ifo, vals_dict, waveform_generator, i_params, releps=1e-4):
        """
            Calculate the partial derivatives of a function at a set of values. The
            derivatives are calculated using the central difference, using an iterative
            method to check that the values converge as step size decreases.

            Parameters
            ----------
            vals_dict: array_like
                A set of values, that are passed to a function, at which to calculate
                the gradient of that function
            i_params: int
                Indices of parameters with respect to which derivatives are to be calculated
            releps: float, array_like, 1e-3
                The initial relative step size for calculating the derivative.

            Returns
            -------
            grads: array_like
                An array of gradients for each non-fixed value.
            """
        
        vals = list(vals_dict.values())
        keys = vals_dict.keys()


        # set steps
        if isinstance(releps, float):
            eps = np.abs(vals) * releps
            eps[eps == 0.] = releps  # if any values are zero set eps to releps
        elif isinstance(releps, (list, np.ndarray)):
            if len(releps) != len(vals):
                raise ValueError("Problem with input relative step sizes")
            eps = np.multiply(np.abs(vals), releps)
            eps[eps == 0.] = np.array(releps)[eps == 0.]
        else:
            raise RuntimeError("Relative step sizes are not a recognised type!")

        ## waveform as function of parameters
        def waveform(parameters):
            polarizations = waveform_generator.frequency_domain_strain(parameters)
            return ifo.get_detector_response(polarizations, parameters)
                    
        i = i_params[0]

        # initial parameter diffs
        leps = eps[i]

        # get central finite difference
        fvals = np.copy(vals)
        bvals = np.copy(vals)

        # central difference
        fvals[i] += 0.5 * leps  # change forwards distance to half eps
        bvals[i] -= 0.5 * leps  # change backwards distance to half eps

        fvals_dict = dict(zip(keys, fvals))
        bvals_dict = dict(zip(keys, bvals))

        grads = (waveform(fvals_dict) - waveform(bvals_dict)) / leps
        
        if len(i_params)>1:
            for i in i_params[1:]:
                # initial parameter diffs
                leps = eps[i]

                # get central finite difference
                fvals = np.copy(vals)
                bvals = np.copy(vals)

                # central difference
                fvals[i] += 0.5 * leps  # change forwards distance to half eps
                bvals[i] -= 0.5 * leps  # change backwards distance to half eps

                fvals_dict = dict(zip(keys, fvals))
                bvals_dict = dict(zip(keys, bvals))
                cdiff = (waveform(fvals_dict) - waveform(bvals_dict)) / leps

                grads = np.vstack((grads, cdiff))

        return grads

    def invertMatrixSVD(self, matrix, threshold=1e-16):

        n_comp = len(matrix[:,0])
        
        matrix_norm = np.zeros((n_comp, n_comp))
        for q in range(n_comp):
            for p in range(n_comp):
                matrix_norm[q,p] = matrix[q,p]/np.sqrt(matrix[p,p]*matrix[q,q])

       # print('Normalized matrix: ',matrix_norm)

                
        ## Inverse of matrix using singular value decomposition (SVD).
        """ This method avoids problems with degeneracy or numerical errors.

            For a given matrix X of order m*n we can decompose it in to U, diagS and V^T 
            U and V^T are resultant real Unitary matrices.

            X (m * n) = U (n * n) diagS (n * m) V^T (m * m)

            U : ndarray
            Unitary matrix having left singular vectors as columns.
            diagS : ndarray
                The singular values, sorted in non-increasing order.
            V^T : ndarray
                Unitary matrix having right singular vectors as rows.
            thresh : int
                least value bound.
        """
        [U, diagS, VT] = np.linalg.svd(matrix_norm)
        kVal = np.sum(diagS > threshold)
        iU = np.conjugate(U).T
        iV = np.conjugate(VT).T
        matrix_inverse = (iV[:,0:kVal] @ np.diag(1/diagS[0:kVal]) @ iU[0:kVal,:])
                
        # print('Matrix inverse: ', matrix_inverse)

        ## Normalizing the inverse matrix
        for q in range(n_comp):
            for p in range(n_comp):
                matrix_inverse[q, p] = matrix_inverse[q, p] / np.sqrt(matrix[p, p] * matrix[q, q])
                    
        return matrix_inverse
                            
    def projection_signal(self, IFOs, sampling_frequency, seg_start_time, seg_end_time, t_coalescence, n_seg, N_samples, bestfit_params,
                          waveform_generator, sub_time_series, parameters=None):

        """
        Parameters
        ----------
        IFOs: A list of Interferometer objects
            Initialization of GW interferometer.
        seg_start_time: float
            The GPS start-time of the data for n_seg.
        seg_end_time: float
            The GPS end-time of the data for n_seg.
        t_coalescence : float array
            Array represent a choice of coalescence time for binary signal.
            for information see Injection.py.
        n_seg: int
            number of time segment in total time duration of operation of detector.
        N_samples: int
            Number of samples in each segment
        bestfit_params: dict
            Dictionary of max likelihood  of all best fit parameters.
        waveform_generator: bilby.gw.waveform_generator.WaveformGenerator
            A WaveformGenerator instance using the source model to inject.
        sub_time_series: float array
            Array of residual time series for each detector

        Return:

        proj_time_series: float array
            Time series obtained by projecting the residual time series.

        """

        print('Projection script starting (', process_time(), ')')
        
        # check if Fisher-matrix calculation is contrained to a subset of parameters
        if parameters==None:
            n_params = len(bestfit_params.loc[0])
            i_params = range(n_params)    
        else:
            n_params = len(parameters)
            keys = bestfit_params.loc[0].to_dict().keys()
            print()
            i_params = np.zeros(n_params, dtype=int)
            for k in range(n_params):
                i_params[k] = list(keys).index(parameters[k])
        
        proj_time_series = np.zeros((len(IFOs), N_samples))
        
        icnt = 0
        for ifo in IFOs:
            PSD = ifo.power_spectral_density
            sub_fourier = ifo.strain_data.frequency_domain_strain
            proj_fourier = 0
            
            tcnt = 0
            # loop over all signals
            for index, single_params in bestfit_params.iterrows():

                ## We use t_coalescence from Injection.py script to have the same coalescence time for an injected and subtracted binary signal from the data of the detector.
                t_c = t_coalescence[tcnt]

                single_params['geocent_time'] = t_c

                if seg_start_time < t_c < seg_end_time:

                    projected = True

                    print("Number of Projection signal is :", index)

                    ## First mass needs to be larger than second mass (just to cross check)
                    if single_params['mass_1'] < single_params["mass_2"]:
                        tmp = single_params['mass_1']
                        single_params['mass_1'] = single_params['mass_2']
                        single_params['mass_2'] = tmp
             
                    waveform_derivatives = self.projection_derivatives(ifo, single_params.to_dict(), waveform_generator, i_params)
                    print('Signal derivatives calculated (', process_time(), ')')
                    ## Defining the Fisher matrix
                    fisher_matrix = np.zeros((n_params, n_params))

                    ## Calculation of Fisher matrix
                    for q in range(n_params):                      
                        ## iterate through columns
                        for p in range(q, n_params):
                            prod = bilby.gw.utils.inner_product(waveform_derivatives[q,:], waveform_derivatives[p,:], waveform_generator.frequency_array, PSD)
                            fisher_matrix[q, p] = prod
                            fisher_matrix[p, q] = prod

                    ## Defining the Correlation matrix = Inverse of Fisher Matrix
                    correlation_matrix = self.invertMatrixSVD(fisher_matrix, threshold=1e-14)

                    #print('Fisher Matrix is :', fisher_matrix)
                    #print(np.shape(fisher_matrix))
                    #print('Correlation Matrix is :', correlation_matrix)
                    #print(np.shape(correlation_matrix))

                    np.save(outdir + '/fisher_matrix_' + ifo.name + '_' + str(n_seg) + '_' + str(index) + '.npy', fisher_matrix)             
 
                    ## Calculating the noise projection
                    scalar_product = np.zeros(n_params)
                        
                    for q in range(n_params):                
                        #MIGHT BE THAT sub_fourier NEEDS TO BE SUBSTITUTED BY proj_fourier (I.E., PROJECTING IN SERIES)
                        scalar_product[q] = bilby.gw.utils.inner_product(waveform_derivatives[q,:], sub_fourier, waveform_generator.frequency_array, PSD)

                    # add projections with respect to all signals in the data
                    proj_fourier += np.matmul(np.matmul(correlation_matrix, scalar_product), waveform_derivatives)

                    #print('Scalar Product is ', scalar_product)
                    
                    tcnt += 1
                    
            proj_time_series[icnt, :] = sub_time_series[icnt, :] - bilby.core.utils.infft(proj_fourier, sampling_frequency) 
            
            print('Projection finished for detector', ifo.name,'(', process_time(), ')')
                    
            icnt += 1
                
        print('Projection script finished (', process_time(), ')')

        return proj_time_series

