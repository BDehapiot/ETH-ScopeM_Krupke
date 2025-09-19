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
from bdtools.models import preprocess
from bdtools.models.annotate import Annotate

#%% Inputs --------------------------------------------------------------------

# Path
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")
train_path = Path("data", "train")

# Procedure
annotate = 0
train = 0
predict = 1

# UNet build()
backbone = "resnet18"
activation = "sigmoid"
downscale_factor = 1

# UNet train()
preview = 0
load_name_0 = ""

# preprocess
patch_size = 768
patch_overlap = 0
img_norm = "image"
msk_type = "normal"

# augment
iterations = 300
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
idx = 47
load_name_1 = "model_768_normal_300-103_1"

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

        # Preprocess (overcoming list bug)
        imgs, msks = preprocess(
            imgs, msks=msks,
            img_norm=img_norm,
            msk_type=msk_type, 
            patch_size=patch_size, 
            patch_overlap=patch_overlap,
            )

        # Train
        
        unet = UNet(
            save_name="",
            load_name=load_name_0,
            root_path=Path.cwd(),
            backbone=backbone,
            classes=1,
            activation=activation,
            )
        
        unet.train(
            
            imgs, msks, 
            X_val=None, y_val=None,
            preview=preview,
            
            # Preprocess
            img_norm="none", 
            msk_type="normal", 
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            downscaling_factor=downscale_factor, 
            
            # Augment
            iterations=iterations,
            invert_p=invert_p,
            gamma_p=gamma_p, 
            gblur_p=gblur_p, 
            noise_p=noise_p, 
            flip_p=flip_p, 
            distord_p=distord_p,
            
            # Train
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            metric=metric,
            learning_rate=learning_rate,
            patience=patience,
            
            )
        
#%% Predict -------------------------------------------------------------------

    if predict:
        
        # Path
        paths = list(data_path.rglob("*.lif"))
        
        # Load data
        img, _ = read_lif(paths[idx], 30)
        
        # Normalize images
        img = norm_pct(img)
        
        # Predict
        unet = UNet(load_name=load_name_1)
        t0 = time.time()
        print("predict : ", end="", flush=False)
        prd = unet.predict(img, verbose=0)
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
                    
        # Display
        vwr = napari.Viewer()
        vwr.add_image(
            img, name="img", visible=1,
            blending="additive", opacity=0.75, gamma=0.25, colormap="gray",
            )
        vwr.add_image(
            prd, name="prd", visible=1,
            blending="additive", opacity=1.00, colormap="magma",
            )