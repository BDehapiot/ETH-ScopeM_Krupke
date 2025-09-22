#%% Imports -------------------------------------------------------------------

import time
import shutil
import pickle
from skimage import io
from pathlib import Path

# functions
from functions import read_lif

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet

#%% Inputs --------------------------------------------------------------------

# Paths
img_name = "all" # image name ("all" for batch processing)
# img_name = "new_20250805_13_3.lif" # image name ("all" for batch processing)
model_name = "model_768_normal_300-103_1"
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")

# Parameters
df = 30 # downscaling factor, should be kept at 30 (DL model)

#%% Function(s) ---------------------------------------------------------------

def process(img_path, df):
    
    print(f"process - {path.name}")
    
    # Paths
    dir_path = img_path.parent / img_path.stem
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(exist_ok=True)
    
    # Extract
    t0 = time.time()
    print("load : ", end="", flush=False)
    img, metadata = read_lif(img_path, df) 
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
           
    # Predict
    t0 = time.time()
    print("predict : ", end="", flush=False)
    prd = unet.predict(norm_pct(img), verbose=0)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
                
    # Save
    
    t0 = time.time()
    print("save : ", end="", flush=False)
    
    # Metadata
    with open(str(dir_path / "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    with open(str(dir_path / "metadata.txt"), "w") as f:
        for key, value in metadata.items():
            f.write(f'{key}: {value}\n')

    # Images
    io.imsave(
        dir_path / "img.tif", img.astype("uint16"), 
        check_contrast=False,
        )
    io.imsave(
        dir_path / "prd.tif", prd.astype("float32"), 
        check_contrast=False,
        )
    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s\n")

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    unet = UNet(load_name=model_name)
    if img_name == "all":
        for path in data_path.glob("*.lif"):
            process(path, df)
    else:
        path = data_path / img_name
        process(path, df)
