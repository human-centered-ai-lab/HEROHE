import os

import openslide

from preprocessing.tile_image import tileSlide

slideending = "mrxs"

filename = "1"


'''
datasetpath = "E:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE"
folderpositive = "positive"
foldernegative = "negative"


slide = openslide.OpenSlide(os.path.join(datasetpath, foldernegative, str(filename) + "." + slideending))

print(slide.level_count)
print(slide.dimensions)
print(slide.level_dimensions)
print(slide.level_downsamples)
print(slide.properties)
print(slide.associated_images)
'''

#tileSlide(os.path.join(datasetpath, foldernegative), "E:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE\\tiled", str(filename) + "." + slideending, 0)
tileSlide("C:\work\TMP", "E:\\tiled", str(filename) + "." + slideending, 0)
