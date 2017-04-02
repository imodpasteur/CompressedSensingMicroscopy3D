#-*-coding:utf-8-*-
# Tools to work with compressed sensing libraries. This file includes simple and useful
#+functions in order to get compressed sensing algorithms to work.
# By MW, GPLv3+, Mar 2016

# TODO
# - Create a Sphinx or readthedocs documentation
# - Get something that works.

# ==== Importations
from __future__ import print_function
import math
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec # subplots with different sizes
    from matplotlib.ticker import NullFormatter # matrices with margins
except:
    print('matplotlib was not imported')
import numpy as np
import scipy.stats, scipy.linalg
import sys
from joblib import Parallel, delayed
from libtiff import TIFF, TIFFfile

try:
    from oct2py import octave
    octave.eval("addpath(genpath('/home/maxime/compressed-sensing/3_code/SPIRALTAP/'))")
    octa = True
except Exception:
    octa = False

try:
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import Ridge
except:
    print('sklearn was not imported')
import pySPIRALTAP
import pyCSalgos.BP.l1eq_pd

# ==== Measure & reconstruct
def measure(data, basis, gaussian=0, poisson=0):
    """Function computes the dot product <x,phi>
    for a given measurement basis phi
    
    Args:
    - data (n-size, numpy 1D array): the initial, uncompressed data
    - basis (nxm numpy 2D array): the measurement basis
    
    Returns:
    - A m-sized numpy 1D array to the dot product"""
    data = np.float_(data)
    
    if gaussian!=0 or poisson!=0: # Create the original matrix
        data = np.repeat([data], basis.shape[0], 0)

    if gaussian!=0: # Bruit
        data +=np.random.normal(scale=gaussian, size=data.shape)
    if poisson != 0:
        data = np.float_(np.random.poisson(np.abs(data)))
        
    if gaussian!=0 or poisson!=0:
        return np.diag((data).dot(basis.transpose()))
    else:
        return (data).dot(basis.transpose())

def reconstruct_1Dmagic(measure, basis):
    """Reconstruction with the L1 minimization provided by L1-magic"""
    x0 = basis.transpose().dot(measure)
    #rec_l1 = pyCSalgos.l1min.l1eq_pd(x0, basis, [], measure, 1e-3)
    rec_l1 = pyCSalgos.BP.l1eq_pd.l1eq_pd(x0, basis, [], measure, 1e-3)
    
    return rec_l1

def reconstruct_1Dlasso(measure, basis):
    """Reconstruction with L1 (Lasso) penalization
    the best value of alpha was determined using cross validation
    with LassoCV"""
    rgr_lasso = Lasso(alpha=4e-3)
    rgr_lasso.fit(basis, measure)
    rec_l1 = rgr_lasso.coef_#.reshape(l, l)
    
    return rec_l1

def reconstruct_1Dmaspiral(measure, basis, maxiter=500, verbose=0):
    """Reconstruct using the Matlab version of SPIRALTAP through the Octave 
    pipe `Oct2Py`"""
    if not octa:
        raise ImportError("It seems something went wrong in the import of `Oct2Py` (or something in the Python <-> Octave bindings.")

    ## ==== Parameters
    tau   = 1e-6
    tolerance = 1e-8

    ## ==== Create function handles
    y=measure.reshape((measure.size,1))
    AT = lambda x: basis.transpose().dot(x)
    A = lambda x: basis.dot(x)
    finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)
    
    octave.SPIRALTAP(y,A,tau,
            'maxiter',maxiter,
            'Initialization',finit,
            'AT',AT,
            'miniter',5,
            'stopcriterion',3,
            'tolerance',tolerance, 
            'alphainit',1,
            'alphamin', 1e-30,
            'alphamax', 1e30,
            'alphaaccept',1e30,
            'logepsilon',1e-10,
            'saveobjective',1,
            'savereconerror',1,
            'savecputime',1,
            'savesolutionpath',0,
            'verbose',verbose)

def reconstruct_1Dspiral(measure, basis, maxiter=500, verbose=0, W=[], penalty='canonical', noisetype='gaussian'):
    """Reconstruct using pySPIRALTAP and default parameters"""

    ## ==== Parameters
    tau   = 1e-6
    tolerance = 1e-8

    ## ==== Create function handles
    y=measure#.reshape((measure.size,1))
    AT = lambda x: basis.transpose().dot(x)
    A = lambda x: basis.dot(x)
    finit = y.sum()*AT(y).size/AT(y).sum()/AT(np.ones_like(y)).sum() * AT(y)

    rec_l1 = pySPIRALTAP.SPIRALTAP(y,A,tau,
                                   AT=AT,
                                   W=W,
                                   maxiter=maxiter,
                                   miniter=5,
                                   penalty=penalty,
                                   noisetype=noisetype,
                                   #initialization=finit,
                                   stopcriterion=3,
                                   tolerance=tolerance,
                                   alphainit=1,
                                   alphamin=1e-30,
                                   alphamax=1e30,
                                   alphaaccept=1e30,
                                   logepsilon=1e-10,
                                   verbose=verbose)[0]
    return rec_l1

def reconstruct_parallel3d(img, basis, ncores=8, algo=reconstruct_1Dmagic):
    """ === CAREFUL, DOC IS THE ONE FROM THE 2D RECONSTRUCTION ====
    A simple wrapper to reconstruct 2D images assuming the lines are independent.
    Careful, this wrapper consider the lines as independent reconstructions to be
    performed.

    Args:
    - img (numpy 2d matrix mxN): input compressed matrix to be decompressed
    - basis (numpy 2d matrix mxn): the measurement matrix used to perform the compression
    - ncores (int): number of cores to use in parallel
    - algo (function): a 1D compressed sensing reconstruction function such as `reconstruct_1Dmagic` or `reconstruct_1Dspiral`.

    Returns:
    - out (numpy 2d matrix nxN): the decompressed image
    """
    N = img.shape[0]
    NN = img.shape[1]
    idx = []
    for i in range(NN): ## Create the list of indices
        idx += zip(range(N), [i]*N)
    
    out=np.zeros((N,NN,basis.shape[1]))
    tmp=Parallel(n_jobs=ncores)(delayed(_reconstruct_parallel)(img[i[0],i[1],:], basis, algo=algo) for i in idx)
    
    for (i, j) in enumerate(tmp): ## Put back in an array
        rr = j
        out[idx[i][0], idx[i][1],:]=rr.flatten()
    return out

def reconstruct_parallel2d(img, basis, ncores=8, algo=reconstruct_1Dmagic, scale=True):
    """A simple wrapper to reconstruct 2D images assuming the lines are independent.
    Careful, this wrapper consider the lines as independent reconstructions to be
    performed.

    Args:
    - img (numpy 2d matrix mxN): input compressed matrix to be decompressed
    - basis (numpy 2d matrix mxn): the measurement matrix used to perform the compression
    - ncores (int): number of cores to use in parallel
    - algo (function): a 1D compressed sensing reconstruction function such as `reconstruct_1Dmagic` or `reconstruct_1Dspiral`.
    - scale (bool): tells whether the data should be rescaled before being processed (default: yes)

    Returns:
    - out (numpy 2d matrix nxN): the decompressed image
    """
    N = img.shape[0]
    
    out=np.zeros((N,basis.shape[1]))
    tmp=Parallel(n_jobs=ncores)(delayed(_reconstruct_parallel)(img[i,:], basis, algo=algo, scale=scale) for i in range(N))
    for (i, j) in enumerate(tmp):
        rr = j
        out[i,:]=rr.flatten()
    return out

def reconstruct_2d(img, basis, algo=reconstruct_1Dspiral, chunksize=128, ncores=8):
    """Reconstructs an image at once by stacking all the vectors and processing them at once
    Careful! This seems to fail when using l1-magic for unknown reasons."""
    if chunksize==None:
        b = scipy.linalg.block_diag(*([basis]*img.shape[0])) ## 2D-fy a 1D basis
        m = img.flatten()
        r = algo(m, b)
        return r.reshape(-1, basis.shape[1])
    elif chunksize > 0:
        ## This method recomputes the big matrix for each iteration, this might be optimized.
        chkidx = np.arange(1, img.shape[0]//chunksize)*chunksize
        mm = np.vsplit(img, chkidx)
        r=Parallel(n_jobs=min(len(mm), ncores))(delayed(_rec)(ch, basis, algo=algo) for ch in mm)
        r = np.vstack(r)
        return r
    else:
        raise InputError("Incoherent chunksize, please a positive integer")

def _rec(ch, ba, algo):
    """Helper function for `reconstruct_2d`"""
    return reconstruct_2d(ch, ba, algo=algo, chunksize=None)

    
def _reconstruct_parallel(m, bf, poisson=False, gaussian=0, algo=None, scale=True):
    """Helper function for `reconstruct_parallel2d`"""
    if algo==None:
        raise ValueError("`algo` must be specified")
    if m.sum() == 0:
        return np.zeros(bf.shape[1])
    if scale:
        M = m.max() # Scale the data
        m = m/M
    #r = cstools.reconstruct_1Dmagic(m, bf)*M
    #r = cstools.reconstruct_1Dspiral(m, bf, maxiter=2500)*M
    r = algo(m, bf)
    if scale:
        r = r*M
    return r

# ==== Data generation algorithms
def add_noise(S, nmes, gaussian=0):
    """Applies gaussian and Poisson noise to a measured matrix
    
    Args:
    - S (numpy a rray): the input original signal
    - nmes (int): number of noisy measurements to perform
    - gaussian (float): intensity of the additive gaussian noise

    Returns:
    - Sn (list of numpy arrays): a list of noisy copies of the image
    """
    Sn = []
    for i in range(nmes):
        if gaussian > 0:
            Sn.append(np.abs(np.random.poisson(S))+np.abs(np.random.normal(0,gaussian,S.shape)))
        else:
            Sn.append(np.abs(np.random.poisson(S)))
    return Sn

def generate_1D(n, sparsity, noise=0):
    """This function generates a random vector with a 
    proportion of 1 is given by the sparsity parameter 
    and where an *additive gaussian noise* is added.
    
    Args:
     - n (int): size of the numpy array to generate
     - sparsity (float): fraction of zero-components of the vector (floored to the nearest possible value)
     - noise (float): standard deviation of a gaussian noise
     
     Returns:
     - a nx1 numpy array"""
    
    n_sig = int(math.floor((1-sparsity)*n))
    sd = np.random.random_integers(0,n-1,(n_sig))# signal
    ss = np.zeros((n))
    ss[sd] = 1
    if noise!=0: # Bruit
        sn=np.random.normal(scale=noise, size=(n))
    else:
        sn=np.zeros((n))
    
    return ss+sn

def gaussian_2D(n, mean, sigma):
    """A multivariate gaussian on a n-shaped array"""
    var = scipy.stats.multivariate_normal(mean=mean, cov=[[sigma,0],[0,sigma]])
    xv = []
    yv = []
    for i in range(n[0]):
        for j in range(n[1]):
            xv.append(i)
            yv.append(j)
    return var.pdf(zip(xv, yv)).reshape(n)

def generate_2D(n=(512, 200), gl=[(.4, 100, 1),(10, 10, 10)], I=int(500*(2*3.14)**.5),
                verbose=False):
    """Generates a 2D image with given number of spots of given variance

    Args:
    - n (2-tuple): size (x,z) of the numpy array to generate
    - gl (list of 3-uple): list of spots (variance, number of spots, rescaling intensity)
    - I (int): the global intensity of the signal

    Returns:
    - S (n-shaped array): array with the given spots and variance placed randomly"""

    S = np.zeros(n)
    gls = []
    for i in gl:
        if verbose:
            print ("signal of variance: ", i[0])
        for j in range(i[1]):
            mean = (np.random.randint(0, n[0]), np.random.randint(0, n[1]))
            S += i[2]*gaussian_2D(n, mean, i[0])
    S=np.int_(S*I)
    return S

# ==== Basis generation algorithms
def generate_bernoulli_basis(n,m, alpha):
    """Function generates a basis for random sampling of the 1D 
    data vector.
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements
    - alpha (float): fraction of planes to be measured, parameter of the Bernoulli law"""
    
    return np.random.binomial(1, alpha, size=(m,n))

def generate_expander_basis(n, m, pr=0.8):
    """Function generates an expander graph, based on the following publication and implementation:

    USAGE: Creating a random nonnegative matrix with p rows and m columns, where pr is the probability 
    for a matrix element to be null. Such a random nonnegative matrix was described in the following paper:

    > M. Raginsky, Z. T. Harmany, R. F. Marcia, and R. M. Willett, "Compressed sensing performance bounds under poisson noise," 
    > IEEE Transactions on Signal Processing, vol. 58, no. 8, pp. 3990 - 4002, 2010.
    """
    p=m
    m=n
    z = np.zeros((p,m));
    temp = np.random.rand(p,m);
    z[temp<pr] = -1*np.sqrt((1-pr)/pr)
    z[temp>=pr] = np.sqrt(pr/(1-pr))

    phi = z/np.sqrt(p);
    phi = phi * np.sqrt(pr*(1-pr)/p) + np.ones((p,m))*(1-pr)/p;
    return phi

def generate_normal_basis(n, m, sigma=1):
    """Function generates a basis for random sampling of the 1D 
    data vector.
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements
    - sigma (float): variance of the normal law"""
    
    return np.random.normal(1, sigma, size=(m,n))

def generate_uniform_basis(n,m):
    """Function generates a basis for random sampling of the 1D 
    data vector.
    
    Args:
    - n (int): size of the data
    - m (int): number of measurements"""
    
    return np.random.uniform(size=(m,n))

def generate_fourier_basis(n,m, positive=True, sample=True, oldmethod=False):
    """Function generates a basis of cosines
    
    Args: 
    - n (int): size of the data
    - m (int): number of measurements
    - positive (bool): tells if we should make sure that the all basis is positive (between 0 and 1)
    
    Returns:
    - a (mxn) matrix.
    """
    if not oldmethod:
        ff = np.fft.fft(np.eye(n))
        re = 0.5*(ff.real[:m,:]+1)
        im = 0.5*(ff.imag[:m,:]+1)
        idx = np.random.randint(2, size=m)
        out = np.zeros((m,n))
        for (i, ii) in enumerate(idx):
            out[i,:] = (re, im)[ii][i,:]
        return out
    
    out = np.zeros((n,n))
    ran = np.array(range(n))
    out[0,:]=1 #0
    i=0
    while 2*i+1 < n:
        out[2*i+1,:]=np.sin(2*np.pi*ran*(i+1)/(n-1))
        if 2*i+2 >= n:
            break
        out[2*i+2,:]=np.cos(2*np.pi*ran*(i+1)/(n-1))
        i+=1
    if sample:
        out = out[np.random.choice(range(n), m, replace=False),:]
    else:
        out = out[0:(m-1),:]
    return (out+1)/2. ## Here we can create photons

# ==== utilities
def read_tif(fn, div2 = False, oldmethod=False, offset2=False):
    """Reads a 3D TIFF file and returns a 3D numpy matrix
    from a path
    
    Inputs :
    - fn (string): the path of the 3D TIFF file
    - div2 (bool): if `True`, only one every to planes in $z$ will be loaded

    Returns :
    - a 3D numpy matrix
    """
    if offset2:
        kp = 1
    else:
        kp = 0
    if not oldmethod:
        ti = TIFFfile(fn)
        ti = ti.get_samples()[0][0].swapaxes(0,2).swapaxes(0,1)
        if div2:
            I = []
            for i in range(ti.shape[2]):
                j = ti[:,:,i]
                if i%2 == kp:
                    I.append(j)
            ti = np.zeros((I[0].shape[0], I[0].shape[1], len(I)))
            for (i,j) in enumerate(I):
                ti[:,:,i]=j
        return ti
    else: # Kept for compatibility. Fails to load some 16 bits images.
        im   = TIFF.open(fn)
        I = []
        for (j,i) in enumerate(im.iter_images()):
            if div2 and (j+1)%2==0:
                pass
            else:
                I.append(i)
        ret = np.zeros((I[0].shape[0], I[1].shape[1], len(I)))
        for (i,ii) in enumerate(I):
            ret[:,:,i]=ii
            return ret

def build_2dmodel(b, psf_xutil):
    """Builds a 2D model based on a measurement matrix for a 1D signal and a PSF 
    following an orthogonal coordinate

    Inputs:
    - b: a mxn numpy array, the measurement matrix
    - psf_xutil: a numpy array representing the 1D PSF.
    """
    
    nrep = len(psf_xutil)
    nrep_pos = int(nrep/2)
    B = np.zeros((b.shape[0]*nrep*3, b.shape[1]*nrep))
    l = []
    for i in psf_xutil:
        l.append(b*i)
    l = np.vstack(l)

    (bs0, bs1) = b.shape
    for i in range(nrep):
        B[(i*bs0):(i*bs0+bs0*nrep),(i*bs1):((i+1)*bs1)]=l
    B=B[(bs0*nrep_pos):(bs0*(nrep+nrep_pos)),:]
    return B
        
def save_tiff(fn, im):
    """Saves a list of images into a TIFF file"""
    raise NotImplementedError()
        
# ==== Plotting functions
def plot_1D(data, sparsity="NA", noise="NA", measured=False):
    """Function plots the output from `generate_1D`. No parameters so far except the data
    
    Args:
    - data (numpy 1D array): output of `generate_1D`.
    
    Returns: None"""
    #plt.figure(figsize=(16, 4))
    plt.plot(data)
    plt.xlim((0,data.shape[0]))
    plt.title("Random data sample \n(sparsity: {}, noise: {})".format(sparsity, noise))
    plt.ylabel("Intensity (normalized)")
    if not measured:
        plt.xlabel("z (slice number)")
    else:
        plt.xlabel("measurement on the basis")

def plot_basis_old(basis, alpha="NA"):
    """Function plots the output from `generate_random_basis`. No parameters so far except the basis itself
    
    Args:
    - data (numpy 1D array): output of `generate_random_basis`.
    - alpha (float): for legend only.
    
    Returns: None"""
    
    #plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 2,width_ratios=[5,1])
    plt.subplot(gs[0])
    
    #plt.subplot(121)
    plt.imshow(basis, cmap=plt.cm.gray, interpolation='none')
    plt.title("Random basis \n(alpha: {})".format(alpha))
    plt.ylabel("Measurement index")
    plt.xlabel("z (slice number)")

    plt.subplot(gs[1])
    b_s = basis.shape
    sx=basis.sum(axis=1)
    plt.plot(sx,range(b_s[0]))
    plt.xlim((0,b_s[1]))
    plt.title("Number of illuminated planes")
    plt.xlabel("number of illuminated planes")
    
def plot_basis(basis):
    """Function plots the output from `generate_random_basis`. No parameters so far except the basis itself
    
    Args:
    - data (numpy 1D array): output of `generate_random_basis`.
    
    Returns: None"""

    nullfmt = NullFormatter()

    # definitions for the axes
    left, width = 0.12, 0.60
    bottom, height = 0.08, 0.60
    bottom_h =  0.16 + width 
    left_h = left - .63 
    rect_plot = [left_h, bottom, width, height]
    rect_x = [left_h, bottom_h, width, 0.2]
    rect_y = [left, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(2, figsize=(16, 5))

    axplot = plt.axes(rect_plot)
    axy = plt.axes(rect_y)

    # Plot the matrix
    axplot.pcolor(basis,cmap=plt.cm.gray)
    plt.ylabel("measurement")

    axplot.set_xlim((0, basis.shape[1]))
    axplot.set_ylim((0, basis.shape[0]))

    axplot.tick_params(axis='both', which='major', labelsize=10)

    # Plot time serie vertical
    axy.plot(basis.sum(axis=1),range(basis.shape[0]),color='k')
    axy.set_xlim((0,basis.shape[1]))
    axy.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel("number of illuminated planes")

def compare_1D(data, recon):
    """Function compares the original data and its reconstruction through Lasso penalization (L1)
    
    Args:
    - data (numpy 1D array): original data.
    - recon (numpy 1D array): reconstructed data (same length as data)
    
    Returns:
    - mean error between `data` and `recon`"""
    #plt.figure(figsize=(16, 8))
    plt.subplot(311)
    plt.plot(recon)
    plt.title("Reconstruction")
    
    plt.subplot(312)
    plt.plot(data)
    plt.title("Data")

    plt.subplot(313)
    plt.plot(np.abs(data-recon))
    plt.title("Absolute rror")
    plt.ylim((data.min(),data.max()))

    return np.abs(recon-data).mean()
