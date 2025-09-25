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
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_dilation
from skimage.morphology import remove_small_holes, remove_small_objects

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Krupke\data")

# Parameters
brush_size = 10

#%% Function(s) ---------------------------------------------------------------

def get_mask(prd):
    msk = prd > 0.5
    msk = remove_small_holes(msk, area_threshold=4096)
    msk = remove_small_objects(msk, min_size=4096)
    return (msk * 255).astype("uint8") 

def get_outline(msk, pnt):
    msk, pnt = msk > 0, pnt > 0
    out = msk ^ binary_erosion(msk)
    out = out & ~pnt
    tmp_out = binary_dilation(out, footprint=disk(2))
    for props in regionprops(label(tmp_out)):
        coords = props.coords
        vals = label(pnt)[tuple(coords.T)]
        if len(np.unique(vals)) != 3:
            out[tuple(coords.T)] = 0
    return (out * 255).astype("uint8")  
    
#%% Class : Correct() ---------------------------------------------------------

class Correct:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.idx = 0
        self.init_data()
        self.init_viewer()
        self.init_layers()
        self.get_info()
        
        # Timers
        self.next_brush_size_timer = QTimer()
        self.next_brush_size_timer.timeout.connect(self.next_brush_size)
        self.prev_brush_size_timer = QTimer()
        self.prev_brush_size_timer.timeout.connect(self.prev_brush_size)
        
    def init_data(self):        
        
        self.imgs, self.prds = [], []
        self.msks, self.pnts, self.outs = [], [], []
        self.img_paths = list(data_path.glob("**/*img.tif"))
        for img_path in self.img_paths:
            
            # Paths
            prd_path = Path(str(img_path).replace("img", "prd"))
            msk_path = Path(str(img_path).replace("img", "msk"))
            pnt_path = Path(str(img_path).replace("img", "pnt"))
            out_path = Path(str(img_path).replace("img", "out"))
            
            # Load
            img = io.imread(img_path)
            prd = io.imread(prd_path)
            self.imgs.append(img)
            self.prds.append(prd)
            
            if msk_path.exists():
                self.msks.append(io.imread(msk_path))
                self.pnts.append(io.imread(pnt_path))
                self.outs.append(io.imread(out_path))
            else:
                self.msks.append(get_mask(prd))
                self.pnts.append(np.zeros_like(img, dtype="uint8"))
                self.outs.append(np.zeros_like(img, dtype="uint8"))
        
        self.imgs = [norm_pct(norm_gcn(img)) for img in self.imgs]
    
    def init_viewer(self):

        self.viewer = napari.Viewer()
        
        # Create "Actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next image")
        self.btn_prev_image = QPushButton("Previous image")
        self.btn_save_changes = QPushButton("Save changes")
        self.btn_revert_changes = QPushButton("Revert changes")
        act_group_layout.addWidget(self.btn_next_image)
        act_group_layout.addWidget(self.btn_prev_image)
        act_group_layout.addWidget(self.btn_save_changes)
        act_group_layout.addWidget(self.btn_revert_changes)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_prev_image.clicked.connect(self.prev_image)
        self.btn_save_changes.clicked.connect(self.save_changes)
        self.btn_revert_changes.clicked.connect(self.revert_changes)
        
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
        
    def init_layers(self):
        
        self.viewer.add_image(
            self.imgs[0].copy(), name="img", visible=1, 
            gamma=0.25,
            )
        self.viewer.add_image(
            self.prds[0].copy(), name="prd", visible=0, 
            colormap="magma",
            )
        self.viewer.add_labels(
            self.msks[0].copy(), name="msk", visible=1,
            blending="translucent", opacity=0.50,
            )
        self.viewer.add_labels(
            self.pnts[0].copy(), name="pnt", visible=1,
            blending="translucent", opacity=0.50, 
            )
        self.viewer.add_image(
            self.outs[0].copy(), name="out", visible=1,
            blending="additive",
            )
        self.viewer.reset_view()
        
        self.viewer.layers["msk"].brush_size = brush_size
        self.viewer.layers["msk"].mode = "paint"
        self.viewer.layers["msk"].selected_label = 255
        self.viewer.layers["pnt"].brush_size = brush_size
        self.viewer.layers["pnt"].mode = "paint"
        self.viewer.layers["pnt"].selected_label = 254
        self.viewer.layers.selection.active = self.viewer.layers["msk"]

#%% Shortcuts -----------------------------------------------------------------
              
        # Buttons
    
        @self.viewer.bind_key("PageDown", overwrite=True)
        def previous_image_key(viewer):
            self.prev_image()
        
        @self.viewer.bind_key("PageUp", overwrite=True)
        def next_image_key(viewer):
            self.next_image()
            
        @Labels.bind_key("Enter", overwrite=True)
        def save_changes_key(viewer):
            self.save_changes() 
            
        @self.viewer.bind_key("Backspace", overwrite=True)
        def revert_changes_key(viewer):
            self.revert_changes() 

        # Keyboard

        @self.viewer.bind_key("Shift", overwrite=True)
        def switch_layer_key(viewer):
            self.viewer.layers.selection.active = self.viewer.layers["pnt"]
            yield
            self.viewer.layers.selection.active = self.viewer.layers["msk"]
        
        @self.viewer.bind_key("0", overwrite=True)
        def pan_switch_key0(viewer):
            self.pan()
            yield
            self.erase()
            
        @self.viewer.bind_key("Space", overwrite=True)
        def pan_switch_key1(viewer):
            self.pan()
            yield
            self.erase()
            
        @self.viewer.bind_key("Right", overwrite=True)
        def next_brush_size_key(viewer):
            self.next_brush_size() 
            # time.sleep(125 / 1000) 
            self.next_brush_size_timer.start(30) 
            yield
            self.next_brush_size_timer.stop()
        
        @self.viewer.bind_key("Left", overwrite=True)
        def prev_brush_size_key(viewer):
            self.prev_brush_size() 
            # time.sleep(125 / 1000) 
            self.prev_brush_size_timer.start(30) 
            yield
            self.prev_brush_size_timer.stop()
            
        @self.viewer.bind_key("Delete", overwrite=True)
        def hide_layers_switch_key(viewer):
            self.hide_layers()
            yield
            self.show_layers()
            
        # Mouse
            
        @self.viewer.mouse_drag_callbacks.append
        def mouse_actions(viewer, event):
            if event.button == 2:
                self.erase()
                yield
                self.paint()
    
#%% Function(s) Shortcuts -----------------------------------------------------
            
    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self.update_layers()
        
    def next_image(self):
        if self.idx < len(self.imgs) - 1:
            self.idx += 1
            self.update_layers()
        
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
    
    def show_layers(self):
        self.viewer.layers["msk"].visible = True
        self.viewer.layers["pnt"].visible = True
        self.viewer.layers["out"].visible = True
    
    def hide_layers(self):
        self.viewer.layers["msk"].visible = False
        self.viewer.layers["pnt"].visible = False
        self.viewer.layers["out"].visible = False
        
#%% Function(s) Actions -------------------------------------------------------

    def update_layers(self):
        self.viewer.layers["img"].data = self.imgs[self.idx]
        self.viewer.layers["msk"].data = self.msks[self.idx]
        self.viewer.layers["pnt"].data = self.pnts[self.idx]
        self.viewer.layers["out"].data = self.outs[self.idx]
        self.viewer.reset_view()
        self.get_info()
        
    def save_changes(self):
        path = str(self.img_paths[self.idx])
        msk = self.viewer.layers["msk"].data.astype("uint8")
        pnt = self.viewer.layers["pnt"].data.astype("uint8")
        if np.max(label(pnt)) != 2:
            raise ValueError("Cannot generate 'out', please check 'pnt' layer")
        out = get_outline(msk, pnt)
        self.viewer.layers["out"].data = out
        self.msks[self.idx] = msk
        self.pnts[self.idx] = pnt
        self.outs[self.idx] = out
        
        # Save
        io.imsave(path.replace("img", "msk"), msk, check_contrast=False)
        io.imsave(path.replace("img", "pnt"), pnt, check_contrast=False)
        io.imsave(path.replace("img", "out"), out, check_contrast=False)

    def revert_changes(self):
        msk = get_mask(self.prds[self.idx])
        pnt = np.zeros_like(msk, dtype="uint8")
        out = np.zeros_like(msk, dtype="uint8")
        self.viewer.layers["msk"].data = msk
        self.viewer.layers["pnt"].data = pnt
        self.viewer.layers["out"].data = out
        self.msks[self.idx] = msk
        self.pnts[self.idx] = pnt
        self.outs[self.idx] = out
        
        # Delete
        path = str(self.img_paths[self.idx])
        Path(path.replace("img", "msk")).unlink()
        Path(path.replace("img", "pnt")).unlink()
        Path(path.replace("img", "out")).unlink()
        
#%% Function(s) Info ----------------------------------------------------------
    
    def get_info(self):
                           
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
            
            f"<span{style2}>- Next/Prev image {spacer * 2}:</span>"
            f"<span{style3}> Page[Up/Down]</span><br>"
            
            f"<span{style2}>- Paint/Erase {spacer * 6}:</span>"
            f"<span{style3}> Mouse[left/Right]</span><br>"
            
            f"<span{style2}>- Save changes {spacer * 5}:</span>"
            f"<span{style3}> Enter</span><br>"  
            
            f"<span{style2}>- Revert changes {spacer * 3}:</span>"
            f"<span{style3}> Backspace</span><br>"
            
            f"<span{style2}>- Change brush size {spacer * 0}:</span>"
            f"<span{style3}> Arrow[Left/Right] </span><br>"
            
            f"<span{style2}>- Pan image {spacer * 8}:</span>"
            f"<span{style3}> Space or Num[0]</span><br>" 
            
            f"<span{style2}>- Hide layers {spacer * 6}:</span>"
            f"<span{style3}> Delete</span><br>"  
            
            )    

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    Correct(data_path)