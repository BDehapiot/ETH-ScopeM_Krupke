#%% Imports -------------------------------------------------------------------

import gc
import numpy as np
from skimage import io
from pathlib import Path
from readlif.reader import LifFile

# bdtools
from bdtools import norm_gcn, norm_pct

# Skimage
from skimage.transform import downscale_local_mean

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Krupke\data")
train_path = Path(Path.cwd(), "data", "train")
df = 30

#%% Function(s) ---------------------------------------------------------------

def extract(path, df):
    
    lif = LifFile(path)
    img_list = [img for img in lif.get_iter_image()]
    
    for i, item in enumerate(img_list):
        
        # Open & rescale image
        pixSize = 1 / item.info["scale"][0] # µm/pixel
        img = np.uint16(item.get_frame(z=0, t=0, c=0))
        img = downscale_local_mean(img, df)
        
        # # Normalize image
        # img = norm_pct(norm_gcn(img), pct_low=0.01, pct_high=99.99)
        
        # Save
        io.imsave(
            str(path).replace(".lif", f"_pix({pixSize * df:.3f})_{i:02d}.tif"),
            img.astype("float32"), check_contrast=False,
            )
        
        del img
        gc.collect()

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for i, path in enumerate(list(data_path.glob("**/*.lif"))):        
        extract(path, df)
