#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path
from readlif.reader import LifFile

# bdmodel
from bdmodel.predict import predict

# Skimage
from skimage.transform import downscale_local_mean
from skimage.morphology import (
    remove_small_holes, remove_small_objects, 
    disk, binary_erosion, white_tophat
    )

#%% Inputs --------------------------------------------------------------------

# Paths
img_name = "all" # image name ("all" for batch processing)
# img_name = "20240709-7_1 merged_new.lif" # image name ("all" for batch processing)
data_path = Path("D:\local_Krupke\data")
model_mass_path = Path.cwd() / "model_mass_normal_768"

# Parameters
df = 30 # downscale factor (should be kept at 30)
max_bin = 1000 # max bin distance in µm
num_bins = 100 # number of bins between 0 and max bin

#%% Function(s) ---------------------------------------------------------------

def extract(path, df, save=False, dir_path=None):    
    lif = LifFile(path)
    img_list = [img for img in lif.get_iter_image()]
    item = img_list[0] # Get first image only
    pixel_size = 1 / item.info["scale"][0] # µm/pixel
    img = np.uint16(item.get_frame(z=0, t=0, c=0))
    img = downscale_local_mean(img, df)
    return img, pixel_size

def get_mask(prd):
    msk = prd > 0.5
    msk = remove_small_holes(msk, area_threshold=4096)
    msk = remove_small_objects(msk, min_size=4096)
    return msk

def clear_borders(out, width=0.01):
    nY, nX = out.shape
    y0, y1 = int(nY * width), int(nY - nY * width)
    x0, x1 = int(nX * width), int(nX - nX * width)
    out[:y0, ...] = 0 ; out[y1:, ...] = 0
    out[..., :x0] = 0 ; out[..., x1:] = 0
    return out

def get_outline(msk):
    out = msk ^ binary_erosion(
        msk, footprint=disk(1), mode="min") 
    out = clear_borders(out, width=0.02)
    return out

def process(img_path, df):
    
    # Paths
    dir_path = img_path.parent / img_path.stem
    dir_path.mkdir(exist_ok=True)
    
    # Extract
    img, pixel_size = extract(img_path, df)    
    
    # Predict
    prd = predict(img, model_mass_path, img_norm="image")
    
    # Get mask & out
    msk = get_mask(prd)
    out = get_outline(msk)
    
    # Subtract background
    tph = white_tophat(img, footprint=disk(21))
    
    # Save
    metadata = {
        "df" : df,
        "pixel_size": pixel_size,
        "pixel_size_df": pixel_size * df,
        }
    
    with open(str(dir_path / "metadata.txt"), "w") as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')
            
    with open(str(dir_path / "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    io.imsave(
        dir_path / "image.tif", img.astype("uint16"), 
        check_contrast=False,
        )
    io.imsave(
        dir_path / "prediction.tif", prd.astype("float32"), 
        check_contrast=False,
        )
    io.imsave(
        dir_path / "mask.tif", (msk * 255).astype("uint8"), 
        check_contrast=False,
        )
    io.imsave(
        dir_path / "outline.tif", (out * 255).astype("uint8"), 
        check_contrast=False,
        )
    io.imsave(
        dir_path / "tophat.tif", tph.astype("float32"), 
        check_contrast=False,
        )
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    if img_name == "all":
        for path in data_path.glob("*.lif"):
            process(path, df)
    else:
        path = data_path / img_name
        process(path, df)