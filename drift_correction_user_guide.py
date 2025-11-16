# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 13:01:17 2025

@author:  Alexandros Papagiannakis, Stanford University, 2025
"""

import remove_image_drift as drft
      
ndtwo_path = ".../images.nd2"
tif_directory = "/tif_images" # folder with exported .tif images. To export images use the "https://github.com/alexSysBio/NDtwoPy" repository

# Open the fifth XY position and the Phase channel form the .nd2 file
images_dict = drft.load_image_arrays(ndtwo_path, 5, 'Phase')
# or
images_dict = drft.load_tif_files(tif_directory)

# Use a single frame to simulate drift
sim_x, sim_y, sim_images_dict = simulate_drift(images_dict[100], 100, 500, 5, "/simulated_images")

# Calculate image drift
phase_drift, cum_drift = drft.generate_drift_sequence(sim_images_dict, resolution=1, None)

# Get the cumulative drift and smooth
# Univariate spline smoothing 
aligned, _, _ = apply_phase_correction(sim_images_dict, cum_drift[0], cum_drift[1], "nearest", ["spline", 2, 50]) # kappa and s parameters 
# or polynomial fit smoothing
aligned, _, _ = apply_phase_correction(sim_images_dict, cum_drift[0], cum_drift[1], "nearest", ["poly", 50]) # polynomial degree
# or moving average smoothing
aligned, _, _ = apply_phase_correction(sim_images_dict, cum_drift[0], cum_drift[1], "nearest", ["rolling", 10]) # rolling window
# or no smoothing
aligned, _, _ = apply_phase_correction(sim_images_dict, cum_drift[0], cum_drift[1], "nearest", None)


# plot the corrected frames
crop_pad = (100,100,1500,1500) # minx, miny, maxx, maxy
scale = 0.066  # um/px
time_stamp_pos = (40,60) # in pixels
scale_bar_pos = (50,110) # in pixels
time_interval= 2 # min
fonts_sizes = 12
fonts_color = 'white'
save_path = "...\Drift_corrected_images"


drft.create_movies(aligned, crop_pad, time_interval, scale, 
                   time_stamp_pos, scale_bar_pos, fonts_sizes, fonts_color, save_path, show=False)

