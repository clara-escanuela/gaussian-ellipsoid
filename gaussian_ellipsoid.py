import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table, join, Column
import scipy
from scipy import stats
import scipy.optimize as opt
import astropy.units as u
from scipy.signal import filtfilt
from scipy.stats import multivariate_normal, skewnorm, norm
from matplotlib.patches import Circle, Ellipse, RegularPolygon

from ctapipe.io import EventSource
from ctapipe.calib import CameraCalibrator
from ctapipe.visualization import CameraDisplay
from ctapipe.image import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image import (hillas_parameters,leakage_parameters,concentration_parameters,)
from ctapipe.image import timing_parameters
from ctapipe.image.morphology import brightest_island, number_of_islands
from ctapipe.image import camera_to_shower_coordinates
from ctapipe.image.cleaning import dilate
from ctapipe.image.toymodel import Gaussian
from ctapipe.image.pixel_likelihood import chi_squared
from ctapipe.utils import linalg

directory = "/home/nieves/Notebooks/ctapipe_analysis/Data"
simtel_url = directory + "/gamma_20deg_0deg_run1555___cta-prod6-paranal-2147m-Paranal-dark-bs5.0-10k-lfa64.simtel.zst"

def chisquar(z, xi, yi, obs, ped_std, emf=1.5):
    """
    Bivariate gaussian distribution
    
    z = [xc, yc, theta, length, width]
    
    """
    
    aligned_covariance = np.array([[z[3] ** 2, 0], [0, z[4] ** 2]])
   
    rotation = linalg.rotation_matrix_2d(z[2]*u.rad)
    
    rotated_covariance = rotation @ aligned_covariance @ rotation.T
  
    prediction = multivariate_normal(allow_singular=True,
            mean=[z[0], z[1]], cov=rotated_covariance
        ).pdf(np.column_stack([xi, yi]))
    
    unc = np.sqrt(ped_std**2 + emf**2*prediction)

    chi_square = (obs - prediction) ** 2 / (unc)**2

    return np.sum(chi_square)

def fitting(cleaned_image, geometry, hillas, ped_std):
    """
    initial_pars = [hillas_centroid_x, hillas_centroid_y, hillas_psi, hillas_length, hillas_width]
    """
    
    xi = geometry.pix_x.value
    yi = geometry.pix_y.value
    x_cent = hillas.x.value
    y_cent = hillas.y.value
    orient = hillas.psi.value
    l = hillas.length.value
    w = hillas.width.value
    
    x0 = [x_cent, y_cent, orient, l, w]
    
    bnds = ((-2, 2), (-2, 2), (-3.15, 3.15), (0, 2), (0, 2))
    
    result = opt.minimize(chisquar, x0=x0, args=(xi, yi, cleaned_image, ped_std), method='SLSQP', bounds=bnds)
    
    return result.x, l

def compute_profile(x, y, nbin=(100,100)):

    # use of the 2d hist by numpy to avoid plotting
    h, xe, ye = np.histogram2d(x,y,nbin)

    # bin width
    xbinw = xe[1] - xe[0]
    x_array      = []
    x_slice_mean = []
    x_slice_rms  = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        if yvals.size>0: # do not fill the quanties for empty slices
            x_array.append(xe[i]+ xbinw/2)
            x_slice_mean.append( yvals.mean())
            x_slice_rms.append( yvals.std())
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_rms = np.array(x_slice_rms)

    return x_array, x_slice_mean, x_slice_rms


gauss_par = []
impact_distance = []
images = []
hillas_length = []

with EventSource(simtel_url) as source:

    core_thr = 6

    subarray = source.subarray
    calibrator = CameraCalibrator(source.subarray)

    for event in source:

        calibrator(event)

        for tel_id in subarray.get_tel_ids_for_type("MST_MST_FlashCam"):

            true_pe = event.simulation.tel[tel_id].true_image

            if true_pe is None:
                continue

            waveforms = event.r1.tel[tel_id].waveform
            ped_mean = waveforms[:, 19:24].mean(axis=-1)
            ped_std = waveforms[:, 19:24].std(axis=-1)
            broken_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]
            dl1 = event.dl1.tel[tel_id]
            geometry = source.subarray.tel[tel_id].camera.geometry

            clean = tailcuts_clean(geometry, dl1.image, picture_thresh=core_thr,
                           boundary_thresh=core_thr/2, min_number_picture_neighbors=1)

            n_islands, labels = number_of_islands(geometry, clean)  #islands

            if n_islands > 0:
                mask = brightest_island(n_islands, labels, dl1.image)
            else:
                mask = clean
           
            n = 2  #number of pixel rows added after cleaning
            m = mask.copy()
            for ii in range(n):
                m = dilate(geometry, m) 

            cleaned_mask = np.array((m.astype(int) + mask.astype(int)), dtype=bool)

            if np.count_nonzero(cleaned_mask) < 6:
                continue

            camera_geometry_brightest = geometry[cleaned_mask]
            charge_brightest = dl1.image[cleaned_mask]
            camera_geometry_brightest = camera_geometry_brightest[charge_brightest>0]
            charge_brightest = list(filter(lambda x : x > 0, charge_brightest))

            cleaned_image = dl1.image
            cleaned_image[~cleaned_mask] = 0.0
            cleaned_image[cleaned_image<0] = 0.0

            hillas = hillas_parameters(camera_geometry_brightest, charge_brightest)
            best_fit, l = fitting(cleaned_image, geometry, hillas, ped_std)

            images.append(cleaned_image)
            gauss_par.append(best_fit)
            hillas_length.append(l)
            impact_distance.append(event.simulation.tel[tel_id].impact.distance.value)

bins = 100

p_x_res, p_mean_res, p_rms_res = compute_profile(impact_distance, np.array(gauss_par)[:, 3], (bins,bins))
p_x_res_h, p_mean_res_h, p_rms_res_h = compute_profile(impact_distance, np.array(hillas_length), (bins,bins))

plt.figure(figsize=(10, 8))

plt.scatter(p_x_res, p_mean_res, color='blue', s=40, alpha=0.5, label='gaussian reconstruction')
plt.scatter(p_x_res_h, p_mean_res_h, color='orange', s=40, alpha=0.5, label='hillas reconstruction')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Reconstructed shower length [m]", fontsize=14)
plt.xlabel("Impact distance [m]", fontsize=14)
plt.legend(fontsize=12)

plt.savefig('/home/nieves/gaussian-ellipsoid/figures/shower_long_impact.png', bbox_inches='tight')


