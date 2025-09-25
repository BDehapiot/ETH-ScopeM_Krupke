## Usage

### `process.py`
Read `.lif` images from `data_path` folder, downscale, predict and save outputs
in a new folder named accordingly to image name.

- Paths
```bash
- img_name      # str, image name ("all" for batch processing)
- model_name    # str, model name (saved in the repo root)
- data_path     # str, path to folder containing nd2 image(s) to process

```

- Parameters
```bash
- df            # int, downscaling factor, should be kept at 30 (DL model)
```

- Outputs
```bash
- img.tif       # uint16, downscaled image
- prd.tif       # float32, DL segmentation prediction
- metadata.txt  # df, original and downscaled pixel size in µm
- metadata.pkl  # df, original and downscaled pixel size in µm
```

<img src='Montage1_RGB_mod.png' width="430" alt="procedure">

### `correct.py`
Read all processed images from `data_path` and display masks (`msk`) for manual 
correction. Adjust the masks and define two points (`pnt`) to delimit the 
considered surface/outline (`out`). Make sure these two points intersect the 
surface of the mask. When finished, press `Enter` to save the corresponding 
outputs.

- Paths
```bash
- data_path  # str, path to folder containing image(s) to process
```

- Parameters
```bash
- brush_size # int, size in pixel(s) of painting/erasing tool
```

- Outputs
```bash
- msk.tif    # uint8, mask of the tissue after manual correction
- pnt.tif    # uint8, mask marking two points that delimit the considered surface/outline
- out.tif    # uint8, mask of the final considered surface/outline
```

<img src='Napari_clipboard.png' width="860" alt="procedure">

### `analyse.py`
Read all processed images from `data_path` and compute Euclidean distance 
transform `edt` of the considered surface/outline `out` to measure fluorescence
intensities according to the distance from the surface.

- Paths
```bash
- data_path # str, path to folder containing image(s) to process
```

- Parameters
```bash
- max_bin   # int, max bin distance in µm
- num_bins  # int, number of bins between 0 and max_bin
```

- Outputs
```bash
- edt.tif   # float32, Euclidean distance transform of outline_hc
- prf.csv   # intensities (A.U.) according to distance (µm)
- fig.png   # plot of intensities (A.U.) according to distance (µm)
```

<img src='Montage3_RGB_mod.png' width="860" alt="procedure">
