#%% Imports -------------------------------------------------------------------

import gc
import numpy as np
from skimage import io
from pathlib import Path
from readlif.reader import LifFile

# Skimage
from skimage.transform import downscale_local_mean

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Krupke\data")
save_path = Path(Path.cwd(), "data", "train")
df = 30

#%% Function(s) ---------------------------------------------------------------

def extract(path, df, save=False, save_path=None):
    
    lif = LifFile(path)
    img_list = [img for img in lif.get_iter_image()]
    
    for i, item in enumerate(img_list):

        pixSize = 1 / item.info["scale"][0] # µm/pixel
        img = np.uint16(item.get_frame(z=0, t=0, c=0))
        img = downscale_local_mean(img, df)
        
        if save:
            
            save_name = path.stem + f"_pix({pixSize * df:.3f})_{i:02d}.tif"
            
            if save_path == None:
                io.imsave(
                    Path(path, save_name), img.astype("uint16"),
                    check_contrast=False
                    )
            else:
                io.imsave(
                    Path(save_path, save_name), img.astype("uint16"),
                    check_contrast=False
                    )
        
            del img
            gc.collect()
            
        else:
            
            return img, pixSize

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    for i, path in enumerate(list(data_path.glob("**/*.lif"))):
        extract(path, df, save=True, save_path=save_path)