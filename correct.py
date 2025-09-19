#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# bdtools
from bdtools import norm_gcn, norm_pct

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QPushButton, QGroupBox, QVBoxLayout, QWidget, QLabel)

# Skimage
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_erosion

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")

# Parameters
brush_size = 10

#%% Class : Correct() ---------------------------------------------------------

class Correct:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.idx = 0
        self.init_images()
        self.init_viewer()
        self.open_images()
        
        # Timers
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)
        
    def init_images(self):        
        self.img_paths = list(data_path.glob("**/*image.tif"))
        self.imgs, self.msks, self.pnts, self.outs = [], [], [], []
        for path in self.img_paths:
            img = io.imread(path)
            msk = io.imread(str(path).replace("image", "mask"))
            self.imgs.append(img)
            self.msks.append(msk)
            self.pnts.append(np.zeros_like(msk))
            self.outs.append(np.zeros_like(msk))
        self.imgs = [norm_pct(norm_gcn(img)) for img in self.imgs]
    
    def init_viewer(self):
        
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            self.imgs[0].copy(), name="img", visible=1, 
            gamma=0.25,
            )
        self.viewer.add_labels(
            self.msks[0].copy(), name="msk", visible=1,
            opacity=0.50, blending="translucent",
            )
        self.viewer.add_labels(
            self.pnts[0].copy(), name="pnt", visible=1,
            opacity=0.50, blending="translucent",
            )
        self.viewer.add_image(
            self.outs[0].copy(), name="out", visible=1,
            blending="additive",
            )
        self.viewer.layers["msk"].brush_size = brush_size
        self.viewer.layers["msk"].mode = "paint"
        self.viewer.layers["msk"].selected_label = 255
        self.viewer.layers["pnt"].brush_size = brush_size
        self.viewer.layers["pnt"].mode = "paint"
        self.viewer.layers["pnt"].selected_label = 254
        self.viewer.layers.selection.active = self.viewer.layers["msk"]
        
        # Create "Actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next image")
        self.btn_prev_image = QPushButton("Previous image")
        self.btn_save_mask = QPushButton("Save mask")
        self.btn_revert_mask = QPushButton("Revert mask")
        act_group_layout.addWidget(self.btn_next_image)
        act_group_layout.addWidget(self.btn_prev_image)
        act_group_layout.addWidget(self.btn_save_mask)
        act_group_layout.addWidget(self.btn_revert_mask)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_prev_image.clicked.connect(self.prev_image)
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_revert_mask.clicked.connect(self.revert_mask)
        
        # Create text
        self.info_image = QLabel()
        self.info_image.setFont(QFont("Consolas"))
        self.info_short = QLabel()
        self.info_short.setFont(QFont("Consolas"))
        
        # Create layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.act_group_box)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_image)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.info_short)
        
        # Create widget
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.viewer.window.add_dock_widget(
            self.widget, area="right", name="Correct") 
        
#%% Shortcuts -----------------------------------------------------------------

        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @Labels.bind_key("Enter", overwrite=True)
        def save_mask_key(viewer):
            self.save_mask() 
            
        @self.viewer.bind_key("Delete", overwrite=True)
        def revert_mask_key(viewer):
            self.revert_mask() 
            
        @self.viewer.bind_key("Backspace", overwrite=True)
        def hide_layers_key(viewer):
            self.hide_layers()
            yield
            self.show_layers()
            
        @self.viewer.bind_key("Control", overwrite=True)
        def layer_switch_key(viewer):
            self.viewer.layers.selection.active = self.viewer.layers["pnt"]
            yield
            self.viewer.layers.selection.active = self.viewer.layers["msk"]

        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch_key1(viewer):
            self.pan()
            yield
            self.erase()
        
        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.pan()
            yield
            self.erase()
            
        @self.viewer.bind_key("Up", overwrite=True)
        def next_brush_size_key(viewer):
            self.next_brush_size() 
            # time.sleep(125 / 1000) 
            self.next_brush_size_timer.start(30) 
            yield
            self.next_brush_size_timer.stop()
        
        @self.viewer.bind_key("Down", overwrite=True)
        def prev_brush_size_key(viewer):
            self.prev_brush_size() 
            # time.sleep(125 / 1000) 
            self.prev_brush_size_timer.start(30) 
            yield
            self.prev_brush_size_timer.stop()
            
        @self.viewer.mouse_drag_callbacks.append
        def mouse_actions(viewer, event):
            if event.button == 2:
                self.erase()
                yield
                self.paint()
                                
#%% Functions(s) --------------------------------------------------------------

    # Shortcuts

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self.open_images()
        
    def next_image(self):
        if self.idx < len(self.imgs) - 1:
            self.idx += 1
            self.open_images()
            
    def get_outline(self, msk_hc, pnt_hc):
        tmp_out = msk_hc ^ binary_erosion(msk_hc)
        tmp_out = tmp_out & ~pnt_hc
        coords, ints = [], []
        for props in regionprops(label(tmp_out), intensity_image=self.imgs[self.idx]):
            coords.append(props.coords)
            ints.append(props.intensity_mean)
        idx = np.argmax(ints)
        out = np.zeros_like(tmp_out, dtype="uint8")
        out[tuple(coords[idx].T)] = 255
        return out        
    
    def save_mask(self):
        msk_hc = self.viewer.layers["msk"].data
        pnt_hc = self.viewer.layers["pnt"].data
        if np.max(label(pnt_hc)) != 2:
            raise ValueError("The mask was not saved, please check pnt layer")
        out = self.get_outline(msk_hc, pnt_hc)
        self.viewer.layers["out"].data = out
        io.imsave(
            str(self.img_paths[self.idx]).replace("image", "mask_hc"),
            msk_hc.astype("uint8"), 
            check_contrast=False,
            )
        io.imsave(
            str(self.img_paths[self.idx]).replace("image", "point_hc"),
            pnt_hc.astype("uint8"), 
            check_contrast=False,
            )
        
    def revert_mask(self):
        self.viewer.layers["msk"].data = self.msks[self.idx].copy()
        
    def show_layers(self):
        self.viewer.layers["msk"].visible = True
        self.viewer.layers["pnt"].visible = True
        self.viewer.layers["out"].visible = True
    
    def hide_layers(self):
        self.viewer.layers["msk"].visible = False
        self.viewer.layers["pnt"].visible = False
        self.viewer.layers["out"].visible = False
        
    def pan(self):
        name = self.viewer.layers.selection.active.name
        self.viewer.layers[name].mode = "pan_zoom"
        
    def paint(self):
        name = self.viewer.layers.selection.active.name
        self.viewer.layers[name].mode = "paint"
        
    def erase(self):
        name = self.viewer.layers.selection.active.name
        self.viewer.layers[name].mode = "erase"
        
    def prev_brush_size(self):
        if self.viewer.layers["msk"].brush_size > 1:
            self.viewer.layers["msk"].brush_size -= 1
            self.viewer.layers["pnt"].brush_size -= 1
        
    def next_brush_size(self): 
        self.viewer.layers["msk"].brush_size += 1
        self.viewer.layers["pnt"].brush_size += 1

    # Procedure

    def open_images(self):
        self.viewer.layers["img"].data = self.imgs[self.idx].copy()
        self.viewer.layers["msk"].data = self.msks[self.idx].copy()
        self.viewer.layers["pnt"].data = self.pnts[self.idx].copy()
        self.viewer.layers["out"].data = self.outs[self.idx].copy()
        self.get_info_text()
        
    # Text 
    
    def get_info_text(self):
                           
        def set_style(color, size, weight, decoration):
            return (
                " style='"
                f"color: {color};"
                f"font-size: {size}px;"
                f"font-weight: {weight};"
                f"text-decoration: {decoration};"
                "'"
                )

        img_name = self.img_paths[self.idx].parent.name

        font_size = 12
        # Set styles (Titles)
        style0 = set_style("White", font_size, "normal", "underline")
        # Set styles (Filenames)
        style1 = set_style("Khaki", font_size, "normal", "none")
        # Set styles (Legend)
        style2 = set_style("LightGray", font_size, "normal", "none")
        # Set styles (Shortcuts)
        style3 = set_style("LightSteelBlue", font_size, "normal", "none")
        spacer = "&nbsp;"

        self.info_image.setText(
            f"<p{style0}>Image<br><br>"
            f"<span{style1}>{img_name}</span><br>"
            )
        
        self.info_short.setText(
            f"<p{style0}>Shortcuts<br><br>"
            
            f"<span{style2}>- Next/Prev image {spacer * 0}:</span>"
            f"<span{style3}> Page[Up/Down]</span><br>"
            
            f"<span{style2}>- Paint/Erase {spacer * 0}:</span>"
            f"<span{style3}> Mouse left/Right</span><br>"
            
            f"<span{style2}>- Save mask    {spacer * 2}:</span>"
            f"<span{style3}> Enter</span><br>"  
            
            f"<span{style2}>- Revert mask  {spacer * 0}:</span>"
            f"<span{style3}> Delete</span><br>"
            
            f"<span{style2}>- Hide mask    {spacer * 2}:</span>"
            f"<span{style3}> Backspace</span><br>"  
            
            f"<span{style2}>- Pan image       {spacer * 2}:</span>"
            f"<span{style3}> Space or Num[0]</span><br>" 
            
            )    
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    # Correct(data_path)
    
#%% development ---------------------------------------------------------------

    from skimage.morphology import binary_erosion, binary_dilation

    # Paths
    name = "old_20240611-12_2"
    img_path = data_path / name / "image.tif"
    msk_hc_path = data_path / name / "mask_hc.tif"
    pnt_hc_path = data_path / name / "point_hc.tif"
    
    # Load
    img = io.imread(img_path)
    msk_hc = io.imread(msk_hc_path) > 0
    pnt_hc = io.imread(pnt_hc_path) > 0
    
    # Get outline    
    tmp_out = msk_hc ^ binary_erosion(msk_hc)
    tmp_out = tmp_out & ~pnt_hc
    tmp_out = binary_dilation(tmp_out)
    out = np.zeros_like(tmp_out)
    for props in regionprops(label(tmp_out), intensity_image=pnt_hc):
        coords = props.coords
        vals = pnt_hc[tuple(coords.T)]
        if np.sum(vals) == 2:
            out[tuple(coords.T)] = 1
            
    
    # out = get_outline(img, msk_hc, pnt_hc)
    
    # Display
    vwr = napari.Viewer()
    vwr.add_image(msk_hc, visible=0)
    vwr.add_image(pnt_hc, visible=1, colormap="yellow")
    vwr.add_image(tmp_out, visible=1, colormap="magenta", blending="additive")
    vwr.add_image(out, visible=1, colormap="green", blending="additive")
