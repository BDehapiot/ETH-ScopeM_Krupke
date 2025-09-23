#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
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

#%% Function(s) ---------------------------------------------------------------

def _analyse(img, edt, metadata, max_bin=5000, num_bins=500):    
    
    pixel_size, df = metadata["pixel_size (µm)"], metadata["df"]
    y, x = img.ravel(), edt.ravel()
    max_bin_pix = max_bin / (pixel_size * df)
    bins = np.linspace(0, max_bin_pix, num_bins + 1)
    indices = np.digitize(x, bins)
    binned_y = [y[indices == i] for i in range(1, len(bins))]
    values = [arr.mean() if len(arr) > 0 else np.nan for arr in binned_y] # np.nan instead of 0s
    bins *= pixel_size * df
    
    # if baseline_pc != 0:
    #     baseline = np.nanpercentile(values, baseline_pc)
    #     values -= baseline
    # else:
    #     baseline = np.nan
    
    return bins, values

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    # for msk_path in msk_paths:
        
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
    
    # Get edt  
    edt = distance_transform_edt(np.invert(out))
    edt[msk == 0] = 0
    
    # 
    bins, values = _analyse(img, edt, metadata)
    plt.plot(bins[1:], values)
    
    # # Display
    # import napari
    # vwr = napari.Viewer()
    # vwr.add_image(img)
    # vwr.add_image(edt)
