#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from extract import extract
from bdmodel.functions import predict

# bdtools
from bdtools.norm import norm_gcn, norm_pct

# Skimage
from skimage.exposure import adjust_gamma
from skimage.morphology import (
    remove_small_holes, remove_small_objects, 
    binary_erosion, binary_dilation, disk
    )

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Krupke\data")
model_mass_path = Path.cwd() / "bdmodel" / "model_mass_768"
model_surface_path = Path.cwd() / "bdmodel" / "model_surface_768"
# img_name = "240611-12_2 merged.lif"
# img_name = "240611-13_4 merged.lif"
# img_name = "240611-16_4 merged.lif"
img_name = "240611-18_4 merged.lif"
img_path = data_path / img_name

# Parameters
df = 30
max_bin = 50
num_bins = 50

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

def analyse(img, edm):
    y, x = img.ravel(), edm.ravel()
    bins = np.linspace(0, max_bin, num_bins + 1)
    indices = np.digitize(x, bins)
    binned_y = [y[indices == i] for i in range(1, len(bins))]
    values = [arr.mean() if len(arr) > 0 else 0 for arr in binned_y]
    return bins, values

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # Extract
    img, pixSize = extract(img_path, df)
    
    # Predict
    prds_mass = predict(img, model_mass_path)
    prds_surface = predict(img, model_surface_path)
    
    # Process
    msk, outlines, edm = process(prds_mass, prds_surface)
    
    # Analyse
    bins, values = analyse(img, edm)
    bins *= pixSize
    dataframe = pd.DataFrame({
        'Dist. (µm)': bins[:-1],
        'Fluo. Int. (A.U.)': values,
        })
    
    # Plot
    plt.hist(bins[:-1], bins=bins, weights=values, edgecolor='black')
    plt.xlabel("Distance from the surface (µm)")
    plt.ylabel("Fluorescence intensity (A.U.)")
    
    # Display
    display = np.zeros((img.shape[0], img.shape[1], 3))
    img_display = adjust_gamma(norm_pct(norm_gcn(img)), gamma=0.9)
    out_display = binary_dilation(outlines * 1)
    display[..., 0] = img_display
    display[..., 1] = np.maximum(img_display, out_display)
    display[..., 2] = img_display
    
    # Save
    save_path = img_path.parent / img_path.stem
    save_path.mkdir(exist_ok=True)
    
    dataframe.to_csv(save_path / "results.csv", index=False)
    
    plt.savefig(save_path / "results.png")
    plt.close()
    
    io.imsave(
        save_path / "img.tif", 
        img.astype("uint16"), 
        check_contrast=False,
        )
    io.imsave(
        save_path / "mask.tif", 
        (msk * 255).astype("uint8"), 
        check_contrast=False,
        )
    io.imsave(
        save_path / "outlines.tif", 
        (outlines * 255).astype("uint8"), 
        check_contrast=False,
        )
    io.imsave(
        save_path / "edm.tif", 
        edm.astype("float32"), 
        check_contrast=False,
        )
    io.imsave(
        save_path / "display.tif", 
        display.astype("float32"), 
        check_contrast=False,
        )
        
    # Display
    viewer = napari.Viewer()
    viewer.add_image(display)
    