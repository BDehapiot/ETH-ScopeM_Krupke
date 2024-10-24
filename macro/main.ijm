// ----------------------------------------------------------------------------

//open("D:/local_Krupke/data/240611-2_4 merged.tif");
//open("D:/local_Krupke/data/240611-2_4 merged_pix10.tif");
//open("D:/local_Krupke/data/240611-2_4 merged_pix10.roi");
//roiManager("Add");

//open("D:/local_Krupke/data/240611-12_2 merged_pix10.tif");
//open("D:/local_Krupke/data/240611-12_2 merged_pix10.roi");
//roiManager("Add");

// Initialize -----------------------------------------------------------------

run("Fire");
setMinAndMax(0, 500);

// Get info
name = getTitle();
stem = File.nameWithoutExtension;
dir = File.directory;
nX = getWidth();
nY = getHeight();
getPixelSize(unit, pixelWidth, pixelHeight);

// Rescale --------------------------------------------------------------------

rf = pixelWidth / 10; // Parameter
nnX = nX * rf;
nnY  = nY * rf;
run("Size...", 
	"width=" + nnX + " height=" + nnY +
    " depth=1 constrain average interpolation=Bilinear"
    );

nnX = getWidth();
nnY = getHeight();

// Draw surface ---------------------------------------------------------------

setTool("polyline");
run("Line Width...", "line=20");
waitForUser("Draw a surface and then press OK to continue.");
run("Fit Spline");
run("Properties... ", "  width=1");
roiManager("Add");
roiManager("Save", dir + stem + ".roi");

// EDM map --------------------------------------------------------------------

//newImage("EDM", "8-bit black", nnX, nnY, 1);
//roiManager("Select", 0);
//run("Draw", "slice");
//run("Invert");
//run("Distance Map");

