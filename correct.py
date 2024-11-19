#%% Imports -------------------------------------------------------------------

from skimage import io
from pathlib import Path

# Napari
import napari
from napari.layers.labels.labels import Labels

# Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QPushButton, QGroupBox, QVBoxLayout, QWidget, QLabel)

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Krupke\data")

#%% Class : Correct() ---------------------------------------------------------

class Correct:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.idx = 0
        self.init_images()
        self.init_viewer()
        self.open_images()
        
    def init_images(self):        
        self.img_paths = list(data_path.glob("**/*image.tif"))
        self.imgs, self.outs = [], []
        for path in self.img_paths:
            self.imgs.append(io.imread(path))
            self.outs.append(
                io.imread(str(path).replace("image", "outline")))
    
    def init_viewer(self):
        
        # Setup viewer
        self.viewer = napari.Viewer()
        self.viewer.add_image(
            self.imgs[0].copy(), name="image", gamma=0.5, opacity=0.75)
        self.viewer.add_labels(
            self.outs[0].copy(), name="outline", blending="translucent")
        self.viewer.layers["outline"].brush_size = 60
        self.viewer.layers["outline"].mode = "erase"
        
        # Create "Actions" menu
        self.act_group_box = QGroupBox("Actions")
        act_group_layout = QVBoxLayout()
        self.btn_next_image = QPushButton("Next Image")
        self.btn_prev_image = QPushButton("Previous Image")
        self.btn_save_outline = QPushButton("Save Outline")
        self.btn_revert_outline = QPushButton("Revert Outline")
        act_group_layout.addWidget(self.btn_next_image)
        act_group_layout.addWidget(self.btn_prev_image)
        act_group_layout.addWidget(self.btn_save_outline)
        act_group_layout.addWidget(self.btn_revert_outline)
        self.act_group_box.setLayout(act_group_layout)
        self.btn_next_image.clicked.connect(self.next_image)
        self.btn_prev_image.clicked.connect(self.prev_image)
        self.btn_save_outline.clicked.connect(self.save_outline)
        self.btn_revert_outline.clicked.connect(self.revert_outline)
        
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
        def save_outline_key(viewer):
            self.save_outline() 
            
        @self.viewer.bind_key("Delete", overwrite=True)
        def revert_outline_key(viewer):
            self.revert_outline() 
            
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
            
        @self.viewer.bind_key("Backspace", overwrite=True)
        def hide_outline_key(viewer):
            self.hide_outline()
            yield
            self.show_outline()
                
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
            
    def save_outline(self):
        out_hc = self.viewer.layers["outline"].data
        io.imsave(
            str(self.img_paths[self.idx]).replace("image", "outline_hc"),
            out_hc.astype("uint8"), 
            check_contrast=False,
            )
        
    def revert_outline(self):
        self.viewer.layers["outline"].data = self.outs[self.idx].copy()
        
    def pan(self):
        self.viewer.layers["outline"].mode = "pan_zoom"
        
    def erase(self):
        self.viewer.layers["outline"].mode = "erase"
        
    def show_outline(self):
        self.viewer.layers["outline"].visible = True
    
    def hide_outline(self):
        self.viewer.layers["outline"].visible = False  

    # Procedure

    def open_images(self):
        self.viewer.layers["image"].data = self.imgs[self.idx].copy()
        self.viewer.layers["outline"].data = self.outs[self.idx].copy()
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
            
            f"<span{style2}>- Next/Prev Image {spacer * 0}:</span>"
            f"<span{style3}> Page[Up/Down]</span><br>"
            
            f"<span{style2}>- Save Outline    {spacer * 3}:</span>"
            f"<span{style3}> Enter</span><br>"  
            
            f"<span{style2}>- Revert Outline  {spacer * 1}:</span>"
            f"<span{style3}> Delete</span><br>"
            
            f"<span{style2}>- Pan Image       {spacer * 6}:</span>"
            f"<span{style3}> Space or Num[0]</span><br>" 
            
            f"<span{style2}>- Hide Outline    {spacer * 3}:</span>"
            f"<span{style3}> Backspace</span><br>"  

            )    
            
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    Correct(data_path)