"""
This is an adjusted implementation from paper provided by:
Li Q, Sone S, Doi K.
"Selective enhancement filters for nodules, vessels, and airway walls in two- and three-dimensional CT scans."
DOI: 10.1118/1.1581411
URL: https://www.ncbi.nlm.nih.gov/pubmed/12945970
"""


import sklearn.mixture
import numpy as np
from .enhanced_functions import *
import scipy.ndimage


class Pores:    
    """Unsupervised semi-automatical pores extraction algorithm.
    Parameters
    ----------        
    radius_min : scalar
        minimum expected pores' radius 
    radius_max : scalar        
        maximum expected pores' radius
    
    %(output)s
    Returns
    -------
    masks : ndarray
        Returned array of same shape as `input`.
    """
    
    def __init__(self, 
                 radius_min, radius_max,
                 enhans_filters_amount=8, 
                 detect_bright_motes=False,
                 normalize=True, pores_integrity=0,
                 dots_level=1, DNG_level=1, lines_level=.5, 
                 label_backprop=False, no_conditions=False):

        self.radius_min = radius_min
        self.radius_max = radius_max
        self.no_conditions = no_conditions
        self.enhans_filters_amount = enhans_filters_amount
        self.detect_bright_motes = detect_bright_motes
        self.normalize = normalize
        self.dots_level = dots_level
        self.DNG_level = DNG_level
        self.lines_level = lines_level
        self.label_backprop = label_backprop
        
        
    def __call__(self, patients, masks=None):
        return self.evaluate(patients, masks)
        
        
    def evaluate(self, patients, masks=None):
        if masks is None:
            masks = [masks] * patients.shape[0]
        
        self.patients = patients
        self.masks = masks
        
        self.calc_enhs()
        print('Enhs has been computed.')
        self.calc_DNG()
        print('DNG has been computed.')
        
        self.threshold()
        if self.label_backprop:
            self.thresholed_dots = np.asarray([label_backprop(dot, origin) 
                                               for dot, origin in zip(self.thresholed_dots, 
                                                                      self.thresholed_dots_origin)])
        return self.thresholed_dots
            
        
    def calc_enhs(self):
        """
        Computes dots and lines enhancement images of a given `self.patients`, `self.masks`,
        for each of sigmas aquired from `self.radius_min`, `self.radius_max` and `self.enhans_filters_amount`.
        if `self.mask` is None: full image area will be considered as a range of interest. 
        """
        sigmas = get_scales(self.radius_min, self.radius_max, 
                            self.enhans_filters_amount)
        
        self.dots, self.lines = list(zip(*[enhs_maxima(patient, sigmas, mask, 
                                                       no_conditions=self.no_conditions,
                                                       likelihood=False, 
                                                       detect_bright=self.detect_bright_motes) 
                                           for patient, mask in zip(self.patients, self.masks)]))
        self.dots = np.asarray(self.dots)
        self.lines = np.asarray(self.lines)
    
    
    def calc_DNG(self):
        """
        Computes divergence of normalized gradient.
        """
        sigmas = get_scales(2 * self.radius_min, 2 * self.radius_max, 
                            self.enhans_filters_amount)
        self.DNG = np.asarray([dng_maxima(patient, sigmas, mask, 
                                           detect_bright=self.detect_bright_motes) 
                                for patient, mask in zip(self.patients, self.masks)])
    
    
    def threshold(self):
        """
        Threshold enhanced images from outliers through `self.DNG_level` * sigma,
        sigma is computed from renormalized destribution.
        """
        self.normalize_DNG()
        self.normalize_lines()
        self.normalize_dots()
        print('Alls has been normalized.')
        
        self.thresholed_dots = np.zeros(self.dots.shape, dtype=np.bool_)
        if self.label_backprop:
                self.thresholed_dots_origin = np.zeros(self.dots.shape)
                
        for i in range(self.patients.shape[0]):
            self.thresholed_dots[i] = self.dots[i] > self.dots_level
            if self.label_backprop:
                self.thresholed_dots_origin[i] = self.thresholed_dots[i].copy()
                
            self.thresholed_dots[i] &= self.DNG[i] > (self.DNG_mean[i] 
                                                + self.DNG_level * self.DNG_std[i])
            self.thresholed_dots[i] &= (self.lines[i] * self.lines_level) < self.dots[i]
        
    
    def normalize_DNG(self):
        """
        Threshold enhanced images from outliers through `self.DNG_level` * sigma,
        sigma is computed from renormalized DNG destribution.
        """
        self.DNG_mean = list()
        self.DNG_std = list()
        converged_ = True
        for i, dng in enumerate(self.DNG):
            GM = sklearn.mixture.GaussianMixture(n_components=2, 
                                                 covariance_type='spherical')

            dng[dng != 0] = np.sqrt(dng[dng != 0] - dng.min())    
            GM.fit(dng[dng != 0].reshape((-1, 1)))

            converged_ &= GM.converged_
            
            self.DNG_mean.append(GM.means_.max())
            self.DNG_std.append(np.sqrt(GM.covariances_[np.argmax(GM.means_)]))
                
        if converged_:
            print('Gaussian mixture has been converged for all cases')
            
            
            
    def normalize_lines(self):
        """
        Threshold enhanced images from outliers through `self.DNG_level` * sigma,
        sigma is computed from renormalized Lines Enhanced destribution.
        """
        self.lines_mean = list()
        self.lines_std = list()
        for i, line in enumerate(self.lines):
            new_line = np.zeros(line.shape)
            new_line[line != 0] = np.sqrt(line[line != 0] - line[line != 0].min())
            
            lower_bound = np.percentile(new_line[line != 0], 1)
            upper_bound = np.percentile(new_line[line != 0], 99)
            
            self.lines_mean.append(np.median(new_line[line != 0]))
            self.lines_std.append(new_line[(new_line > lower_bound) 
                                           & (new_line < upper_bound)].std())
            
            self.lines[i][line != 0] = (new_line[line != 0] - self.lines_mean[-1]) \
                                  / self.lines_std[-1]
            
            
    def normalize_dots(self):
        """
        Threshold enhanced images from outliers through `self.DNG_level` * sigma,
        sigma is computed from renormalized Dots Enhanced destribution.
        """
        self.dots_mean = list()
        self.dots_std = list()
        for i, dot in enumerate(self.dots):
            new_dot = np.zeros(dot.shape)
            new_dot[dot != 0] = np.log(dot[dot != 0])
            
            lower_bound = np.percentile(new_dot[dot != 0], 1)
            upper_bound = np.percentile(new_dot[dot != 0], 99)
            
            self.dots_mean.append(np.median(new_dot[dot != 0]))
            self.dots_std.append(new_dot[(new_dot > lower_bound) 
                                         & (new_dot < upper_bound)].std())
            
            self.dots[i][dot != 0] = (new_dot[dot != 0] - self.dots_mean[-1]) \
                                / self.dots_std[-1]
                                
    
    def label_backprop(bin_new, bin_old, connectivity=4):
        """
        Consider only those dots which has its traces through all time line.
        i.e. nosie dots which appears only for some frames shall be excluded.
        """
        region = np.zeros(bin_old.shape, np.bool)
        out = cv2.connectedComponentsWithStats(bin_new.astype(np.int8), 
                                               connectivity, 
                                               cv2.CV_32S) 
        coords = (out[3][1:, 1].astype(np.int), 
                  out[3][1:, 0].astype(np.int))

        lmap, _ = scipy.ndimage.label(bin_old, structure=None)
        lids = np.unique(lmap[coords])

        for lid in tqdm(lids):
            region |= lmap == lid

        return np.logical_xor((region == 0), bin_old)
