import qupath.lib.scripting.QP
import qupath.lib.gui.QuPathGUI
import qupath.imagej.plugins.ImageJMacroRunner
import qupath.lib.plugins.parameters.ParameterList
import qupath.lib.roi.*
import qupath.lib.objects.*

clearAllObjects()
def imageData = getCurrentImageData()
def props = imageData.getServer().getPath()
def fdir = props.split('/')[-2]
def fname = props.split('/')[-1].split('.jpg')[0]
print(fname)

setImageType('BRIGHTFIELD_H_E');
def hierarchy = imageData.getHierarchy()
def roi = new RectangleROI(0, 0, imageData.getServer().getWidth(), imageData.getServer().getHeight())
def annotation = new PathAnnotationObject(roi, PathClassFactory.getPathClass("Region"))
hierarchy.addPathObject(annotation, false)
def annotations = hierarchy.getAnnotationObjects()
selectAnnotations();
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.55112 0.7837 0.28651 ", "Stain 2" : "Eosin", "Values 2" : "0.12121 0.85354 0.50674 ", "Background" : " 255 255 255 "}');

runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Hematoxylin OD",  "requestedPixelSizeMicrons": 0.5,  "backgroundRadiusMicrons": 8.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 400.0,  "threshold": 0.3,  "maxBackground": 2.0,  "watershedPostProcess": true,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

detections = getDetectionObjects()
print("I have " + detections.size() + " detections")

def path = buildFilePath(PROJECT_BASE_DIR, "detections", fname + "_" + fdir + ".txt")
saveDetectionMeasurements(path)
