#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Krupke\data")
img_paths = list(data_path.glob("**/*image.tif"))

# Parameters
max_bin = 1000 # max bin distance in µm
num_bins = 100 # number of bins between 0 and max bin

#%% Function(s) ---------------------------------------------------------------

def analyse(img_path, max_bin=1000, num_bins=100):
        
    def _analyse(tph, edt, pixel_size, df, max_bin=max_bin, num_bins=num_bins):    
        y, x = tph.ravel(), edt.ravel()
        max_bin_pix = max_bin / (pixel_size * df)
        bins = np.linspace(0, max_bin_pix, num_bins + 1)
        indices = np.digitize(x, bins)
        binned_y = [y[indices == i] for i in range(1, len(bins))]
        values = [arr.mean() if len(arr) > 0 else 0 for arr in binned_y]
        return bins, values

    # Paths
    dir_path = img_path.parent
    
    # Open metadata       
    with open(str(dir_path / "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Open data
    img_path = str(img_path)
    img = io.imread(img_path)
    msk = io.imread(img_path.replace("image", "mask"))
    tph = io.imread(img_path.replace("image", "tophat"))
    if Path(img_path.replace("image", "outline_hc")).exists():
        out = io.imread(img_path.replace("image", "outline_hc"))
    else:
        out = io.imread(img_path.replace("image", "outline"))
        print(f"outline for {Path(img_path).parent.name} has not been corrected")

    # Get edt  
    edt = distance_transform_edt(np.invert(out))
    edt[msk == 0] = 0
    
    # Analyse
    bins, values = _analyse(tph, edt, metadata["pixel_size"], metadata["df"])
    
    
    
    
    

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    analyse(data_path)
    
    pass
