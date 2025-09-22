#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from pathlib import Path
from readlif.reader import LifFile

# skimage
from skimage.transform import downscale_local_mean


#%% Functions -----------------------------------------------------------------

def read_lif(path, df):    
    lif = LifFile(path)
    img_list = [img for img in lif.get_iter_image()]
    item = img_list[0] # Get first image only
    pixel_size = 1 / item.info["scale"][0] # µm/pixel
    img = np.uint16(item.get_frame(z=0, t=0, c=0))
    img = downscale_local_mean(img, df).astype("uint16")
    metadata = {
        "df" : df,
        "pixel_size (µm)": pixel_size,
        "pixel_size_df (µm)": pixel_size * df,
        }
    return img, metadata

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    import napari
    
    # Paths
    data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")
    lif_paths = list(data_path.rglob("*.lif"))
    
    # Inputs
    df, idx = 30, -1
    
    # Load
    t0 = time.time()
    print("load : ", end="", flush=False)
    img, metadata = read_lif(lif_paths[idx], df)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(img)