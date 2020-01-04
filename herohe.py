import os
import openslide
from preprocessing.tile_image import tileSlide

slideextension = "mrxs"
datasetpath = "G:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE"
folderpositive = "positive"
foldernegative = "negative"
outputpath = "H:\HEROHE_CHALLENGE\DataSet\\tiled"

#filename = "1"

'''
slide = openslide.OpenSlide(os.path.join(datasetpath, foldernegative, str(filename) + "." + slideextension))

print(slide.level_count)
print(slide.dimensions)
print(slide.level_dimensions)
print(slide.level_downsamples)
print(slide.properties)
print(slide.associated_images)
'''

for root, dirs, files in os.walk(datasetpath, topdown=False):
   for name in files:
      if name.endswith(slideextension):
         print(os.path.join(root, name))
         #print(root)
         #print(name)
         tileSlide( root, outputpath, name, 0)

#tileSlide(os.path.join(datasetpath, foldernegative), outputpath, str(filename) + "." + slideextension, 0)
#tileSlide("C:\work\TMP", "E:\\tiled", str(filename) + "." + slideextension, 0)
