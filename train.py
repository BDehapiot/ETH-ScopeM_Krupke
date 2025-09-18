#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path

# functions
from functions import read_lif

# bdtools
from bdtools.norm import norm_pct
from bdtools.models.unet import UNet
from bdtools.models.annotate import Annotate

#%% Inputs --------------------------------------------------------------------

# Path
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")
train_path = Path("data", "train")

# Procedure
annotate = 0
train = 1
predict = 0

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 1
load_name_0 = ""

# preprocess
patch_size = 768
patch_overlap = 0
img_norm = "image"
msk_type = "normal"

# augment
iterations = 500
invert_p = 0.0
gamma_p = 0.5
gblur_p = 0.5
noise_p = 0.5 
flip_p = 0.5 
distord_p = 0.5

# train
epochs = 100
batch_size = 8
validation_split = 0.2
metric = "soft_dice_coef"
learning_rate = 0.001
patience = 20

# predict
cond = 1
idx = 5
load_name_1 = ""

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
#%% Annotate ------------------------------------------------------------------
    
    if annotate:
        Annotate(train_path)
    
#%% Train ---------------------------------------------------------------------
    
    if train:
    
        # Load data
        imgs, msks = [], []
        for path in list(train_path.rglob("*.tif")):
            if "mask" in path.name:
                if Path(str(path).replace("_mask", "")).exists():
                    msks.append(io.imread(path))   
                    imgs.append(io.imread(str(path).replace("_mask", "")))
        # imgs = np.stack(imgs)
        # msks = np.stack(msks)

        # unet = UNet(
        #     save_name="",
        #     load_name=load_name_0,
        #     root_path=Path.cwd(),
        #     backbone=backbone,
        #     classes=1,
        #     activation=activation,
        #     )
        
        # # Train
        # unet.train(
            
        #     imgs, msks, 
        #     X_val=None, y_val=None,
        #     preview=preview,
            
        #     # Preprocess
        #     img_norm=img_norm, 
        #     msk_type=msk_type, 
        #     patch_size=patch_size,
        #     patch_overlap=patch_overlap,
        #     downscaling_factor=downscale_factor, 
            
        #     # Augment
        #     iterations=iterations,
        #     invert_p=invert_p,
        #     gamma_p=gamma_p, 
        #     gblur_p=gblur_p, 
        #     noise_p=noise_p, 
        #     flip_p=flip_p, 
        #     distord_p=distord_p,
            
        #     # Train
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     validation_split=validation_split,
        #     metric=metric,
        #     learning_rate=learning_rate,
        #     patience=patience,
            
        #     )
        
#%% Predict -------------------------------------------------------------------

    # if predict:
        
    #     # Imports
    #     from skimage.filters import gaussian
        
    #     # Path
    #     path = list(data_path.rglob("*.nd2"))[cond] 
        
    #     # Load data
    #     metadata = check_nd2(path)
    #     C0 = read_nd2(path, z=idx, c=0)     
    #     C1 = read_nd2(path, z=idx, c=1)       
        
    #     # Predict edt
    #     unet = UNet(load_name=load_name_edt)
    #     t0 = time.time()
    #     print("predict edt : ", end="", flush=False)
    #     prd_edt = (unet.predict(C1, verbose=0) * 255).astype("uint8")
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")
        
    #     # Predict interfaces
    #     unet = UNet(load_name=load_name_interfaces)
    #     t0 = time.time()
    #     print("predict interfaces : ", end="", flush=False)
    #     prd_interfaces = (unet.predict(C1, verbose=0) * 255).astype("uint8")
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")
        
    #     # Predict bounds
    #     unet = UNet(load_name=load_name_skeletons)
    #     t0 = time.time()
    #     print("predict skeletons : ", end="", flush=False)
    #     prd_skeletons = (unet.predict(C1, verbose=0) * 255).astype("uint8")
    #     t1 = time.time()
    #     print(f"{t1 - t0:.3f}s")
        
    #     # Merge predicitions
    #     prd_merge = prd_edt.astype(float) - (prd_interfaces.astype(float))
    #     prd_merge += prd_skeletons.astype(float)
        
    #     def merge_predictions(
    #             prd_edt, prd_interfaces, prd_skeletons, coeff=9):
    #         prd_edt = prd_edt.astype(float)
    #         prd_interfaces = prd_interfaces.astype(float)
    #         prd_skeletons = prd_skeletons.astype(float)
    #         prd_merge = (
    #             prd_edt - (coeff * prd_interfaces) + (coeff * prd_skeletons))
    #         prd_merge[prd_merge < 0] = 0
    #         prd_merge = norm_pct(prd_merge)
    #         return prd_merge
        
    #     prd_merge = merge_predictions(
    #         prd_edt, prd_interfaces, prd_skeletons, coeff=5)
    #     prd_merge = gaussian(prd_merge, sigma=0.5, preserve_range=True)
                
    #     # ---
            
    #     # Display
    #     vwr = napari.Viewer()
    #     vwr.add_image(
    #         C0, name="C0", visible=0,
    #         blending="additive", opacity=0.75, colormap="gray",
    #         )
    #     vwr.add_image(
    #         C1, name="C1", visible=1,
    #         blending="additive", opacity=0.75, colormap="gray",
    #         )
    #     vwr.add_image(
    #         prd_edt, name="prd_edt", visible=0,
    #         blending="additive", opacity=1.00, colormap="bop blue",
    #         )
    #     vwr.add_image(
    #         prd_interfaces, name="prd_interfaces", visible=0,
    #         blending="additive", opacity=1.00, colormap="bop orange",
    #         )
    #     vwr.add_image(
    #         prd_skeletons, name="prd_skeletons", visible=0,
    #         blending="additive", opacity=1.00, colormap="bop blue",
    #         )
    #     vwr.add_image(
    #         prd_merge, name="prd_merge", visible=1,
    #         blending="additive", opacity=1.00, colormap="magma",
    #         )
        