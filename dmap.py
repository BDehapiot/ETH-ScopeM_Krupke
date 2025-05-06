#%% Imports -------------------------------------------------------------------

import pickle
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.filters import gaussian
from skimage.morphology import disk, binary_dilation

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

#%% Inputs --------------------------------------------------------------------

data_path = data_path = Path("D:\local_Krupke\data")
name = "20240709-4_5 merged"

# Parameters


#%% Execute -------------------------------------------------------------------

if __name__ == "__main__": 

    # Load
    edt = io.imread(data_path / name / "edt.tif")
    out = io.imread(data_path / name / "outline.tif")
    out_hc = io.imread(data_path / name / "outline_hc.tif")
    with open(str(data_path / name / "metadata.pkl"), "rb") as file:
        md = pickle.load(file)
    pixel_size = md["pixel_size_df (µm)"]
    
    # Process images
    edt *= pixel_size
    out = binary_dilation(out)
    out_hc = binary_dilation(out_hc, footprint=disk(5))
    out_hc = gaussian(out_hc, sigma=1)
    
    # Set cmaps
    cmap_g = plt.cm.gray
    cmap_t = cmap_g(np.arange(cmap_g.N))
    cmap_t[:, -1] = np.linspace(0, 1, cmap_g.N)
    cmap_t[0, -1] = 0
    cmap_t = mcolors.ListedColormap(cmap_t)
    cmap_d = plt.cm.viridis(np.linspace(0, 1, 256))
    cmap_d[0] = [0, 0, 0, 1]  # RGBA
    cmap_d = mcolors.ListedColormap(cmap_d)
    
    # Set colorbar
    vmin, vmax = 0, 12000  # manual limits
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_d, norm=norm)
    sm.set_array([])
    
    # Set contours
    contour_levels = np.arange(1000, 12000, 1000)
    
    # Plot
    fig, ax = plt.subplots()
    ax.imshow(edt, cmap=cmap_d, norm=norm)
    ax.contour(edt, levels=contour_levels, colors='white', linewidths=0.5)
    cbar = fig.colorbar(sm, ax=ax)
    ax.imshow(out_hc, cmap=cmap_t)
    
    # Formatting
    
    ax.set_title("Distance from ")
    cbar.set_label("Distance (µm)") 
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save
    plt.tight_layout()
    plt.savefig(
        Path.cwd() / "dmap_plot.svg", format="svg")
    plt.close(fig)    
    