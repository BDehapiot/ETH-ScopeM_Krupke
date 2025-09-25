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

# Parameters
max_bin = 10000  # max bin distance in µm
num_bins = 1000  # number of bins between 0 and max bin

# baseline_pc = 10 # percentage (0 to 100) of lowest values to be considered baseline

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
    edt = edt.astype("float32")
    
    # Get profile
    prf = _get_profile(
        img, edt, metadata, max_bin=max_bin, num_bins=num_bins)
    
    # baseline = np.nanpercentile(prf["intensity"], baseline_pc)
    # prf["intensity_sub"] = prf["intensity"] - baseline

    return edt, prf

def plot_profile(prf, theme="dark"):
    fig, ax = plt.subplots()
    ax.hist(
        prf["distance"], bins=prf["distance"], weights=prf["intensity"],
        color="gray",
        )
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Fluo. intensity (A.U.)")  
    fig.tight_layout()
    plt.close(fig)
    return fig 

def analyse(msk_path):
    
    print(f"analyse - {msk_path.parent.name}")
    
    # Paths 
    mtd_path = Path(str(msk_path).replace("msk.tif", "metadata.pkl"))
    img_path = Path(str(msk_path).replace("msk", "img"))
    out_path = Path(str(msk_path).replace("msk", "out"))
    edt_path = Path(str(msk_path).replace("msk", "edt"))
    prf_path = Path(str(msk_path).replace("msk.tif", "prf.csv"))
    fig_path = Path(str(msk_path).replace("msk.tif", "fig.png"))
    
    # Load
    with open(str(mtd_path), "rb") as f:
        metadata = pickle.load(f)
    img = io.imread(img_path)
    msk = io.imread(msk_path)
    out = io.imread(out_path)
    
    # Get profile
    edt, prf = get_profile(
        img, msk, out, metadata, max_bin=max_bin, num_bins=num_bins)
    
    # Plot profile
    fig = plot_profile(prf)
    
    # Save
    io.imsave(edt_path, edt, check_contrast=False)
    prf.to_csv(prf_path, index=False)
    fig.savefig(fig_path, format="png")
     
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    msk_paths = list(data_path.glob("**/*msk.tif"))
    for msk_path in msk_paths:
        analyse(msk_path)