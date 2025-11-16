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
from skimage.io import imread
import os


"""
Application of the nd2_to_array library to load .nd2 images or the Scikit-Image io linrary to load .tif images.
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
Application of the cross correlation function frim Scikit-Image, to calculate  the phase drift between consecutive frames.
"""
def generate_drift_sequence(images_dict, resolution, hard_threshold, mask_show):
    
    frame_n = max(images_dict.keys())
    
    phase_y = []
    phase_x = []
    
    for fr in range(frame_n):
        
        image_before = images_dict[fr]
        image_after =  images_dict[fr+1]
        
        if hard_threshold == 'otsu':
            otsu_before = threshold_otsu(image_before.ravel())
            otsu_after = threshold_otsu(image_after.ravel())
            otsu = np.mean([otsu_before, otsu_after])
        elif type(hard_threshold)==int:
            otsu = hard_threshold
        elif hard_threshold == 'none':
            print('')
        else:
            raise ValueError(f'hard threshold value {hard_threshold} is not valid. Choose "otsu" or a phase contrast integer value. Choose "none" to avoid using binary masks for the drift estimation (default)')
    
        if  type(hard_threshold)==int or hard_threshold == 'otsu':
            phase_dif = phase_cross_correlation(image_before, image_after, 
                                                upsample_factor=resolution,
                                                reference_mask = image_before<otsu,
                                                moving_mask = image_after<otsu)[0]
        elif hard_threshold == 'none':
            phase_dif = phase_cross_correlation(image_before, image_after, 
                                                upsample_factor=resolution)[0]

            
        if mask_show == True:
            plt.imshow((image_before<otsu)+(image_after<otsu))
            plt.show()
        
        print(phase_dif)
        
        phase_x.append(phase_dif[0])
        phase_y.append(phase_dif[1])
        
    return phase_x, phase_y



"""
Methodologies to smooth the drifts
"""
def rolling_smooth_dirfts(cum_phase_drift, window):
    
    drift_df = pd.DataFrame()
    drift_df['cum_drift_x'] = cum_phase_drift[0]
    drift_df['cum_drift_y'] = cum_phase_drift[1]
    mean_df = drift_df.rolling(window, min_periods=1, center=True).mean()
    
    return mean_df.cum_drift_x, mean_df.cum_drift_y

def poly_smooth_dirfts(cum_phase_drift, degree):
    
    cum_phase_x = cum_phase_drift[0]
    cum_phase_y = cum_phase_drift[1]
    
    fit_x = np.polyfit(np.arange(len(cum_phase_x)), cum_phase_x, degree)
    fit_y = np.polyfit(np.arange(len(cum_phase_y)), cum_phase_y, degree)

    cum_phase_x = np.polyval(fit_x,np.arange(len(cum_phase_x)))
    cum_phase_y = np.polyval(fit_y, np.arange(len(cum_phase_y)))
    
    return cum_phase_x, cum_phase_y

def univar_smooth_dirfts(cum_phase_drift, kappa, smoothing):
    
    cum_phase_x = cum_phase_drift[0]
    cum_phase_y = cum_phase_drift[1]
    
    fit_x = UnivariateSpline(np.arange(len(cum_phase_x)), cum_phase_x, k=kappa, s=smoothing)
    fit_y = UnivariateSpline(np.arange(len(cum_phase_y)), cum_phase_y, k=kappa, s=smoothing)

    cum_phase_x = fit_x(np.arange(len(cum_phase_x)))
    cum_phase_y = fit_y(np.arange(len(cum_phase_y)))
    
    return cum_phase_x, cum_phase_y



"""
Subtraction of the smoothed cumulative phase drift using a cropping frame.
"""
def apply_phase_correction(images_dict, phase_dif, smooth, smooth_factor):
    
    cum_phase_x = np.cumsum(phase_dif[0])
    cum_phase_y = np.cumsum(phase_dif[1])
    
    plt.figure()
    plt.plot(cum_phase_x, 'o', color='black', label ='x drift')
    plt.plot(cum_phase_y, 'o', color='gray', label = 'y drift')
    
    if smooth == 'rolling':
        cum_phase_x, cum_phase_y = rolling_smooth_dirfts((cum_phase_x, cum_phase_y), smooth_factor)
        plt.plot(cum_phase_x, color='red', label ='moving av. x drift')
        plt.plot(cum_phase_y, color='red', label = 'moving av. y drift')
    elif smooth == 'poly':
        cum_phase_x, cum_phase_y = poly_smooth_dirfts((cum_phase_x, cum_phase_y), smooth_factor)
        plt.plot(cum_phase_x, color='red', label ='poly x drift')
        plt.plot(cum_phase_y, color='red', label = 'poly y drift')
    elif smooth == 'univar':
        cum_phase_x, cum_phase_y = univar_smooth_dirfts((cum_phase_x, cum_phase_y), smooth_factor[0], smooth_factor[1])
        plt.plot(cum_phase_x, color='red', label ='univar x drift')
        plt.plot(cum_phase_y, color='red', label = 'univar y drift')
    else:
        print('No smoothing applied. Choose "rolling" for a moving average or "poly" for a polynomial fit.')
    
    plt.legend()
    plt.show()
    
    # min_x = int(np.min(cum_phase_x))
    max_x = int(np.max(np.abs(cum_phase_x)))
    # min_y = int(np.min(cum_phase_y))
    max_y = int(np.max(np.abs(cum_phase_y)))
    
    cor_images_dict = {}
    
    for fr in images_dict:
        
        img = images_dict[fr]
        
        if fr == 0:
            ph_x = 0 
            ph_y = 0
        elif fr > 0:
            ph_x = int(cum_phase_x[fr-1])
            ph_y = int(cum_phase_y[fr-1])
        
        framed_img = img[max_y+ph_y:img.shape[0]-max_y+ph_y, 
                         max_x+ph_x:img.shape[1]-max_x+ph_x]
        
        cor_images_dict[fr] = framed_img
    
    return cor_images_dict, (cum_phase_x, cum_phase_y)



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


    
    




