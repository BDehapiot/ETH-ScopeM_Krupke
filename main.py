#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from extract import extract

# bdmodel
from bdmodel.functions import preprocess
from bdmodel.predict import predict

# bdtools
from bdtools.norm import norm_gcn, norm_pct

# Skimage
from skimage.exposure import adjust_gamma
from skimage.morphology import (
    remove_small_holes, remove_small_objects, 
    disk, binary_erosion, binary_dilation, white_tophat
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Paths
img_name = "20240709-41_5 merged.lif" # image name
data_path = Path("D:\local_Krupke\data")
model_mass_path = Path.cwd() / "model_mass_normal_768"
model_surface_path = Path.cwd() / "model_surface_normal_768"
img_path = data_path / img_name

# Parameters
df = 30 # downscale factor (should be kept at 30)
max_bin = 1000 # max bin distance in µm
num_bins = 100 # number of bins between 0 and max bin

#%% Function(s) ---------------------------------------------------------------

def process(prds_mass, prds_surface):
    msk_mass = prds_mass > 0.5
    msk_mass = remove_small_holes(msk_mass, area_threshold=1024)
    msk_mass = remove_small_objects(msk_mass, min_size=4096)
    outlines = msk_mass ^ binary_erosion(msk_mass, footprint=disk(1), mode="min")    
    msk_surface = prds_surface > 0.5
    outlines[msk_surface == 0] = False
    edm = distance_transform_edt(np.invert(outlines))
    edm[msk_mass == 0] = 0
    return msk_mass, outlines, edm

def analyse(img, edm, pixSize, df, tophat=None):    
    y, x = img.ravel(), edm.ravel()
    max_bin_pix = max_bin / (pixSize * df)
    bins = np.linspace(0, max_bin_pix, num_bins + 1)
    indices = np.digitize(x, bins)
    binned_y = [y[indices == i] for i in range(1, len(bins))]
    values = [arr.mean() if len(arr) > 0 else 0 for arr in binned_y]
    return bins, values

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    from bdtools.patch import extract_patches   
    from bdmodel.functions import preprocess

    # Extract
    img, pixSize = extract(img_path, df)    
        
    # Preprocess
    patches = preprocess(
        img, msks=None, 
        img_norm="image",
        patch_size=768, 
        patch_overlap=0,
        )
    
    '''
    
    - something is wrong with one patch prediction!
    
    '''
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(patches)
    # viewer.add_image(prds_mass)
    # viewer.add_image(prds_surface)
    
    # # Predict
    # prds_mass = predict(img, model_mass_path, img_norm="image")
    # prds_surface = predict(img, model_surface_path, img_norm="image")
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(img)
    # viewer.add_image(prds_mass)
    # viewer.add_image(prds_surface)
    
    # # Process
    # msk, outlines, edm = process(prds_mass, prds_surface)
    
    # # Analyse
    # img = white_tophat(img, footprint=disk(21)) # Subtract background
    # bins, values = analyse(img, edm, pixSize, df)
    # bins *= pixSize * df
    # dataframe = pd.DataFrame({
    #     'Dist. (µm)': bins[:-1],
    #     'Fluo. Int. (A.U.)': values,
    #     })
    
    # # Plot
    # plt.hist(bins[:-1], bins=bins, weights=values, edgecolor='black')
    # plt.xlabel("Distance from the surface (µm)")
    # plt.ylabel("Fluorescence intensity (A.U.)")
    
    # # Display
    # display = np.zeros((img.shape[0], img.shape[1], 3)).astype("uint8")
    # img_display = adjust_gamma(norm_pct(norm_gcn(img)), gamma=0.5)
    # out_display = binary_dilation(outlines * 1)
    # img_display = (img_display * 255).astype("uint8")
    # out_display = (out_display * 255).astype("uint8")
    # display[..., 0] = img_display
    # display[..., 1] = np.maximum(img_display, out_display)
    # display[..., 2] = img_display
    
    # # Save
    # save_path = img_path.parent / img_path.stem
    # save_path.mkdir(exist_ok=True)
    
    # dataframe.to_csv(save_path / "results.csv", index=False)
    
    # plt.savefig(save_path / "results.png")
    # plt.close()
    
    # io.imsave(
    #     save_path / "img.tif", 
    #     img.astype("uint16"), 
    #     check_contrast=False,
    #     )
    # io.imsave(
    #     save_path / "mask.tif", 
    #     (msk * 255).astype("uint8"), 
    #     check_contrast=False,
    #     )
    # io.imsave(
    #     save_path / "outlines.tif", 
    #     (outlines * 255).astype("uint8"), 
    #     check_contrast=False,
    #     )
    # io.imsave(
    #     save_path / "edm.tif", 
    #     edm.astype("float32"), 
    #     check_contrast=False,
    #     )
    # io.imsave(
    #     save_path / "display.tif", 
    #     display, 
    #     check_contrast=False,
    #     )
        
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(display)
    