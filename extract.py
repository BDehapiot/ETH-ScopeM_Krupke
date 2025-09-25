#%% Imports -------------------------------------------------------------------

import time
from skimage import io
from pathlib import Path
from functions import read_lif

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")
train_path = Path(Path.cwd(), "data", "train")
lif_paths = list(data_path.rglob("*.lif"))

# Parameters
df = 30

#%% Function(s) ---------------------------------------------------------------

def extract(path, df):
    
    print(f"extract - {path.name}")
    
    # Load
    t0 = time.time()
    print("load : ", end="", flush=False)
    img, _ = read_lif(path, df)
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    # Save
    t0 = time.time()
    print("save : ", end="", flush=False)
    io.imsave(
        train_path / (path.stem + f"_df-{df}.tif"),
        img, check_contrast=False,
        )    
    t1 = time.time()
    print(f"{t1 - t0:.3f}s")
    
    pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in lif_paths:
        extract(path, df)