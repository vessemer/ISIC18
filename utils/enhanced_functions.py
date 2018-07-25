
from tqdm import tqdm
import numpy as np
import scipy.ndimage
import scipy


def get_scales(bottom=4, top=16, 
               amount=6):
    radius = (top / bottom) ** (1. / (amount - 1))
    sigmas = [bottom / 4.]
    for i in range(amount - 1):
        sigmas.append(sigmas[0] * (radius ** i + 1))
    return sigmas


def apply_enhs_filters(patient, sigma, mask=None,
                       likelihood=False, detect_bright=True,
                       no_conditions=False):
    filtered = scipy.ndimage.filters.gaussian_filter(patient, 
                                                     sigma=sigma)
    if mask is None:
        mask = np.ones(patient.shape)
    grad = np.gradient(filtered * mask)

    axis = [[0, 1], [1]]
    hess = [np.gradient(deriv, axis=j) 
            for i, deriv in enumerate(grad) 
            for j in axis[i]]

    #   [(0, xx), (1, xy), (2, yy)]
    #   x, y -> 2, 2, x, y -> 2, 2, (x * y)
    
    coords = np.where(mask)
    for j in range(len(hess)):
        hess[j] = hess[j][coords]

    K = .5 * (hess[0] + hess[2])
    Q = np.sqrt(np.abs(hess[0] * hess[2] - hess[1] * hess[1]))
    eigs = np.asarray([K + np.sqrt(np.abs(K * K - Q * Q)),
                       K - np.sqrt(np.abs(K * K - Q * Q))])

    old = eigs.copy()
    eigs = np.sort(np.abs(eigs), axis=0) == np.abs(old)
    eigs = np.where(eigs.sum(axis=0))[0]
    old = old.T
    old[eigs] = old[eigs, ::-1]
    eigs = old
    
    if detect_bright:
        condition1 = (eigs[:, 0] < 0).astype(np.float)
        condition2 = (eigs[:, 1] < 0).astype(np.float)
    else:
        condition1 = (eigs[:, 0] > 0).astype(np.float)
        condition2 = (eigs[:, 1] > 0).astype(np.float)
    
    enh_dot =  np.abs(eigs[:, 1]) / np.abs(eigs[:, 0])
    enh_line = np.abs(eigs[:, 0]) - np.abs(eigs[:, 1])
    
    if likelihood:
        enh_line /= np.abs(eigs[:, 0])
    else:
        enh_dot *= (sigma ** 2) * np.abs(eigs[:, 1])
        enh_line *= (sigma ** 2)
        
    if not no_conditions:
        enh_dot *= (condition1 * condition2)
        enh_line *= condition1
        
    return enh_dot, enh_line, coords


def dng(sigma, patient):
    grad = np.asarray(np.gradient(patient))
    grad /= scipy.linalg.norm(grad, axis=0) + 1e-3
    grad = [scipy.ndimage.filters.gaussian_filter(deriv, 
                                                  sigma=sigma) 
            for deriv in grad]
    return np.sum([np.gradient(el, axis=i) 
                   for i, el in enumerate(grad)], 
                  axis=0)


def dng_maxima(patient, sigmas, mask=None, 
               detect_bright=True):
    divergences = list()

    for sigma in sigmas:
        divergences.append(dng(patient=patient, 
                               sigma=sigma))
        
    
    if mask is not None:
        divergences *= mask
    if detect_bright:
        divergences = -1 * np.asarray(divergences)
    return divergences.max(axis=0)


def enhs_maxima(patient, sigmas, mask=None, no_conditions=False,
                likelihood=False, detect_bright=True):
    enhs_dot = list()
    enhs_line = list()
    z_dot = np.zeros(patient.shape)
    z_line = np.zeros(patient.shape)

    for sigma in sigmas:
        enh_dot, enh_line, coords = apply_enhs_filters(patient=patient, 
                                                       mask=mask, 
                                                       no_conditions=no_conditions,
                                                       sigma=sigma, 
                                                       likelihood=likelihood, 
                                                       detect_bright=detect_bright)
        enhs_dot.append(enh_dot)
        enhs_line.append(enh_line)

    z_dot[coords] = np.asarray(enhs_dot).max(axis=0)
    z_line[coords] = np.asarray(enhs_line).max(axis=0)
    return z_dot, z_line
