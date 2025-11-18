# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 10:10:22 2025

@author: Alexandros Papagiannakis, Stanford University, 2025
"""

from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nd2_to_array as ndt
from scipy.interpolate import UnivariateSpline
from skimage.io import imread, imsave
import os
import time
import pickle


"""
Application of the nd2_to_array library to load .nd2 images or the scikit-image io linrary to load .tif images.
"""
def load_image_arrays(ndtwo_path, xy_position, channel):
    
    return ndt.nd2_to_array(ndtwo_path)[2][xy_position-1][channel]


def load_tif_files(tif_directory):
    img_dict= {}
    i = 0
    for img_path in os.listdir(tif_directory):
        print(img_path)
        img_dict[i] = imread(tif_directory+'/'+img_path)
        i+=1
    return img_dict
        


"""
Generate drift to check the code
"""
def get_time_string(frame, digits=4):
    return (4-len(str(frame)))*'0'+str(frame)



def simulate_drift(image_array, number_of_frames, padding, drift_std, save_path):
    
    x_drift = np.random.normal(loc=0, scale=drift_std, size=number_of_frames)
    y_drift = np.random.normal(loc=0, scale=drift_std, size=number_of_frames)
    
    drift_dict = {}
    padding = int(padding)
    H, W = image_array.shape
    crop_pad = np.array([padding, W-padding+1, padding, H-padding+1]) #minx, maxxx, miny, maxy
    drift_dict[0] = image_array[crop_pad[2]:crop_pad[3], crop_pad[0]:crop_pad[1]]
    imsave(save_path+'/'+get_time_string(0)+'frame.tif', drift_dict[0])
    
    i = 1
    for xd, yd in zip(x_drift, y_drift):
        drift_array = np.array([int(xd), int(xd), int(yd), int(yd)])
        crop_pad = crop_pad + drift_array
        drift_dict[i] = image_array[crop_pad[2]:crop_pad[3], crop_pad[0]:crop_pad[1]]
        imsave(save_path+'/'+get_time_string(i)+'frame.tif', drift_dict[i])
        i+=1
        
    return x_drift, y_drift, drift_dict



"""
Application of the cross correlation function frim Scikit-Image, to calculate  the phase drift between consecutive frames.
"""
def generate_drift_sequence(images_dict, resolution, hard_threshold):
    
    start_time = time.time()
    
    frame_keys = sorted(images_dict.keys())
    if len(frame_keys) < 2:
        raise ValueError("Need at least two timepoints to compute drift.")
    
    phase_y = [0.0]
    phase_x = [0.0]
    cum_y = [0.0]
    cum_x = [0.0]
    
    for pre_k, next_k in zip(frame_keys[:-1], frame_keys[1:]):
        
        image_before = images_dict[pre_k]
        image_after =  images_dict[next_k]
        
        use_masks = False
        ref_mask = None
        mov_mask = None
        
        if isinstance(hard_threshold, (int, np.integer)):
            thr = int(hard_threshold)
            use_masks = True
            ref_mask = image_before < thr
            mov_mask = image_after < thr
    
        elif isinstance(hard_threshold, str) and hard_threshold.lower() == 'otsu':
            thr_before = threshold_otsu(image_before)
            thr_after = threshold_otsu(image_after)
            thr = (thr_before + thr_after) / 2.0
            use_masks = True
            ref_mask = image_before < thr
            mov_mask = image_after < thr
    
        elif isinstance(hard_threshold, str) and hard_threshold.lower() == 'none':
            use_masks = False
    
        else:
            raise ValueError(
                f'hard_threshold value {hard_threshold} is not valid. '
                'Choose "otsu", an integer, or "none" (default) to avoid using masks.'
            )
        
        
        shift, _, _ = phase_cross_correlation(
            image_before,
            image_after,
            upsample_factor=resolution,
            reference_mask=ref_mask if use_masks else None,
            moving_mask=mov_mask if use_masks else None
        )
    
        # print(shift)
        phase_x.append(float(shift[1]))
        phase_y.append(float(shift[0]))
        cum_x.append(cum_x[-1]+shift[1])
        cum_y.append(cum_y[-1]+shift[0])
        
        if pre_k%100 == 0:
            current_time = time.time()
            print(f'Computing drift: {pre_k} out of {len(frame_keys)} positions, {cum_x[-1]}, {cum_y[-1]}, {current_time-start_time:.2f} sec')
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return (np.array(phase_x), np.array(phase_y)), (np.array(cum_x), np.array(cum_y))



"""
Methodologies to smooth the drifts
"""
def rolling_smooth_drifts(cum_phase_drift, window):
    
    drift_df = pd.DataFrame()
    drift_df['cum_drift_x'] = cum_phase_drift[0]
    drift_df['cum_drift_y'] = cum_phase_drift[1]
    mean_df = drift_df.rolling(window, min_periods=1, center=True).mean()
    
    return mean_df.cum_drift_x, mean_df.cum_drift_y

def poly_smooth_drifts(cum_phase_drift, degree):
    
    cum_phase_x = cum_phase_drift[0]
    cum_phase_y = cum_phase_drift[1]
    
    fit_x = np.polyfit(np.arange(len(cum_phase_x)), cum_phase_x, degree)
    fit_y = np.polyfit(np.arange(len(cum_phase_y)), cum_phase_y, degree)

    cum_phase_x = np.polyval(fit_x,np.arange(len(cum_phase_x)))
    cum_phase_y = np.polyval(fit_y, np.arange(len(cum_phase_y)))
    
    return cum_phase_x, cum_phase_y

def univar_smooth_drifts(cum_phase_drift, kappa, smoothing):
    
    cum_phase_x = cum_phase_drift[0]
    cum_phase_y = cum_phase_drift[1]
    
    fit_x = UnivariateSpline(np.arange(len(cum_phase_x)), cum_phase_x, k=kappa, s=smoothing)
    fit_y = UnivariateSpline(np.arange(len(cum_phase_y)), cum_phase_y, k=kappa, s=smoothing)

    cum_phase_x = fit_x(np.arange(len(cum_phase_x)))
    cum_phase_y = fit_y(np.arange(len(cum_phase_y)))
    
    return cum_phase_x, cum_phase_y


def apply_smoothing(cum_x, cum_y, smooth_params):
    
    fit_x =cum_x[1:]
    fit_y = cum_y[1:]
    
    if isinstance(smooth_params, list):
        if smooth_params[0].lower() in ("rolling", "window", "mean"):
            xs, ys = rolling_smooth_drifts((fit_x, fit_y), smooth_params[1])
    
        elif smooth_params[0].lower() in ("poly", "polynomial"):
            xs, ys = poly_smooth_drifts((fit_x, fit_y), smooth_params[1])
    
        elif smooth_params[0].lower() in ("spline", "univariate", "univar"):
            xs, ys = univar_smooth_drifts((fit_x, fit_y), smooth_params[1], smooth_params[2])

        else:
            raise ValueError("Unknown smoother. Use 'rolling', 'poly', 'spline', a callable, or None.")
    elif smooth_params is None:
            xs, ys = fit_x, fit_y
    else:
        raise ValueError("smoother must be a string, callable, or None.")
    return [0]+list(xs), [0]+list(ys)


"""
Subtraction of the smoothed cumulative phase drift using a cropping frame.
"""
def apply_phase_correction(images_dict, cum_x, cum_y, rounding, smooth_params):
    
    keys = sorted(images_dict.keys())
    if len(keys) == 0:
        return {}, None, (None, None)
    
    # Validate shapes
    shapes = [images_dict[k].shape for k in keys]
    H, W = shapes[0][0], shapes[0][1]
    for s in shapes:
        if s[0] != H or s[1] != W:
            raise ValueError("All images must have the same height and width.")
    
    plt.plot(cum_x)
    plt.plot(cum_y)
    cum_x, cum_y = apply_smoothing(cum_x, cum_y, smooth_params)
    plt.plot(cum_x)
    plt.plot(cum_y)
    plt.show()
    cum_x = np.asarray(cum_x, dtype=float)
    cum_y = np.asarray(cum_y, dtype=float)
    if cum_x.shape[0] != len(keys) or cum_y.shape[0] != len(keys):
        raise ValueError("cum_x and cum_y must match number of frames (including leading 0).")
    
    # Quantize to integer-pixel shifts (cropping-only alignment)
    if rounding == "nearest":
        ix = np.rint(cum_x).astype(int)
        iy = np.rint(cum_y).astype(int)
    elif rounding == "floor":
        ix = np.floor(cum_x).astype(int)
        iy = np.floor(cum_y).astype(int)
    elif rounding == "ceil":
        ix = np.ceil(cum_x).astype(int)
        iy = np.ceil(cum_y).astype(int)
    else:
        raise ValueError("rounding must be 'nearest', 'floor', or 'ceil'.")
    
    y0 = int(np.max(iy))
    y1 = int(H + np.min(iy))
    x0 = int(np.max(ix))
    x1 = int(W + np.min(ix))
    
    if y1 <= y0 or x1 <= x0:
        raise ValueError("Drift too large for given image size: no common overlap remains.")
        
    aligned = {}
    for i, k in enumerate(keys):
        img = images_dict[k]
        # Source window in the original image that maps to [y0:y1, x0:x1] in aligned canvas
        sy0 = y0 - iy[i]
        sy1 = y1 - iy[i]
        sx0 = x0 - ix[i]
        sx1 = x1 - ix[i]
        
        aligned[k] = img[sy0:sy1, sx0:sx1]
        
    return aligned, (y0, y1, x0, x1), (ix, iy)
      

def save_drift_statistics(drift_save_path, ix, iy):
    with open(drift_save_path+'/cumulative_drift_lists', 'wb') as handle:
        pickle.dump([ix, iy], handle)
        

"""
Visualization.
"""
def generate_movie_frames(cor_images_dict, save_path):
    
    for fr in cor_images_dict:
        plt.imshow(cor_images_dict[fr], cmap='gray')
        plt.savefig(save_path+'/'+str(fr)+'.jpeg')
        plt.close()
    

def get_time_stamp(time):
    
    hr = int(time/60)
    hr_str = (2-len(str(hr)))*'0'+str(hr)
    mn = time-hr*60
    mn_str = (2-len(str(mn)))*'0'+str(mn)
    
    return hr_str+':'+mn_str


def create_movies(drift_corrected_images_dict, crop_pad, time_interval, scale, 
                  time_stamp_pos, scale_bar_pos, fonts_sizes, fonts_color, save_path, show=False):
    
    drift_cor_images_dict = drift_corrected_images_dict

    for fr in drift_cor_images_dict:
        img = drift_cor_images_dict[fr]
        crop_img = img[crop_pad[1]:crop_pad[3], crop_pad[0]:crop_pad[2]]
        plt.figure(figsize=((crop_pad[2]-crop_pad[0])/100,(crop_pad[3]-crop_pad[1])/100))
        plt.imshow(crop_img, cmap='gray')
        
        time = fr*time_interval #min
        time_stamp = get_time_stamp(time)
        
        plt.text(*time_stamp_pos,time_stamp, fontsize=fonts_sizes, color=fonts_color)
        plt.text(*scale_bar_pos, r'5 $\mu$m', fontsize=fonts_sizes, color=fonts_color)
        plt.plot([scale_bar_pos[0]+80-5/0.066,scale_bar_pos[0]+80],[scale_bar_pos[1]+20,scale_bar_pos[1]+20], color=fonts_color, linewidth=6)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path+'/'+str(fr)+'.jpeg')
        if show == True:
            plt.show()
        else:
            plt.close()


    
    

