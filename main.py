#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path

# Functions
from bdmodel.functions import predict

# Skimage
from skimage.morphology import (
    remove_small_holes, remove_small_objects, 
    binary_erosion, disk
    )

#%% Inputs --------------------------------------------------------------------

# Paths
model_path = Path.cwd() / "bdmodel" /"model_mass_768"
# imgs_path = Path.cwd() / "data" / "240611-12_2 merged_pix(13.771)_00.tif"
# imgs_path = Path.cwd() / "data" / "240611-13_4 merged_pix(13.771)_00.tif"
# imgs_path = Path.cwd() / "data" / "240611-16_4 merged_pix(13.771)_00.tif"
imgs_path = Path.cwd() / "data" / "240611-18_4 merged_pix(13.771)_00.tif"

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    img = io.imread(imgs_path)
    
    # Predict
    prds = predict(        
        img,
        model_path,
        img_norm="global",
        patch_overlap=0,
        )
    
    # 
    msk = prds > 0.5
    msk = remove_small_holes(msk, area_threshold=1024)
    msk = remove_small_objects(msk, min_size=1024)
    outlines = msk ^ binary_erosion(msk, footprint=disk(1), mode="min")
    
    # Display
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_image(prds)
    viewer.add_image(msk)
    viewer.add_image(outlines)
    