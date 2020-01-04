import os
import openslide
from preprocessing.tile_image import tileSlide

slideextension = "mrxs"
datasetpath = "G:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE"
folderpositive = "positive"
foldernegative = "negative"
outputpath = "H:\HEROHE_CHALLENGE\DataSet"

filename = "1"

slide = openslide.OpenSlide(os.path.join(datasetpath, foldernegative, str(filename) + "." + slideextension))

print(slide.level_count)
print(slide.dimensions)
print(slide.level_dimensions)
print(slide.level_downsamples)
print(slide.properties)
print(slide.associated_images)


#tileSlide(os.path.join(datasetpath, foldernegative), "E:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE\\tiled", str(filename) + "." + slideextension, 0)
#tileSlide("C:\work\TMP", "E:\\tiled", str(filename) + "." + slideextension, 0)
