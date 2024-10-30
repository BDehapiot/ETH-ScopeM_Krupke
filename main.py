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

# Skimage
from skimage.filters import gaussian
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
    
    # Plot
    plt.hist(bins[:-1], bins=bins, weights=values, edgecolor='black')
    plt.xlabel("Distance from the surface (µm)")
    plt.ylabel("Fluorescence intensity (A.U.)")
    
    # /////////////////////////////////////////////////////////////////////////
    
    # Save display
    display = []
    display.append(img.astype("float32"))
    display.append((msk * 65535).astype("float32"))
    display.append((outlines * 65535).astype("float32"))
    display.append(edm.astype("float32"))
    display.append(edm.astype("float32"))
    display = np.stack(display)
    
    io.imsave(
        str(img_path).replace(".lif", "_display.tif"),
        display,
        check_contrast=False,
        imagej=True,
        metadata={
            'axes': 'CYX', 
            'mode': 'composite',
            }
        )
        
    # /////////////////////////////////////////////////////////////////////////
    
    # Save dataframe
    dataframe = pd.DataFrame({
        'Dist. (µm)': bins[:-1],
        'Fluo. Int. (A.U.)': values,
        })
    dataframe.to_csv(
        str(img_path).replace(".lif", "_results.csv"), 
        index=False
        )
    
    
    # # Display
    # viewer = napari.Viewer()
    # viewer.add_image(img, gamma=0.5)
    # viewer.add_image(prds_mass)
    # viewer.add_image(prds_surface)
    # # viewer.add_image(msk)
    # viewer.add_image(outlines, blending="additive")
    # viewer.add_image(edm)
    