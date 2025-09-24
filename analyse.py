#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")
msk_paths = list(data_path.glob("**/*msk.tif"))

# Parameters
max_bin = 10000
num_bins = 1000

#%% Function(s) ---------------------------------------------------------------

def get_profile(img, msk, out, metadata, max_bin=10000, num_bins=1000):
    
    def _get_profile(img, edt, metadata, max_bin=max_bin, num_bins=num_bins):    
        y, x = img.ravel(), edt.ravel()
        max_bin_pix = max_bin / metadata["pixel_size_df"]
        bins = np.linspace(0, max_bin_pix, num_bins + 1)
        indices = np.digitize(x, bins)
        binned_y = [y[indices == i] for i in range(1, len(bins))]
        values = [arr.mean() if len(arr) > 0 else np.nan for arr in binned_y]
        values.append(np.nan)
        bins *= metadata["pixel_size_df"]
        prf = np.column_stack((bins, values))
        prf = pd.DataFrame(prf, columns=["distance", "intensity"])
        return prf
    
    # Get edt  
    edt = distance_transform_edt(np.invert(out))
    edt[msk == 0] = 0
    
    # Get profile
    prf = _get_profile(
        img, edt, metadata, max_bin=max_bin, num_bins=num_bins)

    return edt, prf

def plot_profile(prf, theme="dark"):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.hist(
        prf["distance"], bins=prf["distance"], weights=prf["intensity"],
        color="black",
        )
    ax.set_xlabel("Distance (µm)", fontsize=4)
    ax.set_ylabel("Fluo. intensity (A.U.)", fontsize=4)  
    ax.tick_params(axis="both", labelsize=4, width=0.25, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    fig.tight_layout()
    return fig 
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    msk_path = msk_paths[0]
        
    # Paths 
    pkl_path = Path(str(msk_path).replace("msk.tif", "metadata.pkl"))
    img_path = Path(str(msk_path).replace("msk", "img"))
    out_path = Path(str(msk_path).replace("msk", "out")) 
    
    # Load
    with open(str(pkl_path), "rb") as f:
        metadata = pickle.load(f)
    img = io.imread(img_path)
    msk = io.imread(msk_path)
    out = io.imread(out_path)
    
    # Get profile
    edt, prf = get_profile(
        img, msk, out, metadata, max_bin=10000, num_bins=1000)
    
    # Plot profile
    fig = plot_profile(prf)

    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(img)
    # vwr.add_image(edt, visible=0)
    # vwr.add_image(out, blending="additive")
