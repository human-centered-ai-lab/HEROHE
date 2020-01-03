import os
import time
import PIL
import openslide
import matplotlib.pyplot as plt

tileSizeX = 512
tileSizeY = 512
cutoff = tileSizeX*tileSizeY


def getRed(redVal):
   return '#%02x%02x%02x' % (redVal, 0, 0)


def getGreen(greenVal):
   return '#%02x%02x%02x' % (0, greenVal, 0)


def getBlue(blueVal):
   return '#%02x%02x%02x' % (0, 0, blueVal)

def tileSlide(inputpath, outputpath, imagename, level):
   startTime = time.time()
   slide = openslide.OpenSlide(os.path.join(inputpath, imagename))
   endTime = time.time()
   print("Reading file in " + str(endTime - startTime) + " sec.")
   slideWidth, slideHeight = slide.level_dimensions[level ]

   tilesX = int(slideWidth / tileSizeX) + 1
   tilesY = int(slideHeight / tileSizeY) + 1

   print(str(startTime) + ": Tiling slide: " + str(imagename) + " [" + str(slideWidth) + "x" + str(slideHeight) + "] into "
         + str(tilesX) + " Tiles in x and " + str(tilesY) + " Tiles in y.")

   fileExtenstionPosition = imagename.rfind(".")
   outputFolder = os.path.join(outputpath, imagename[:fileExtenstionPosition] + "_" + str(tileSizeX) + "x" + str(tileSizeY))
   if not os.path.exists(outputFolder):
      os.makedirs(outputFolder)

   print("Preparation: " + str(endTime - time.time()))

   for tilePositionX in range(tilesX):
      for tilePositionY in range(tilesY):
         startFor = time.time()
         timer = time.time()
         locationX = tilePositionX * tileSizeX
         locationY = tilePositionY * tileSizeY
         tileWidth = min(slideWidth, locationX + tileSizeX) - locationX
         tileHeight = min(slideHeight, locationY + tileSizeY) - locationY
         tile = slide.read_region((locationX, locationY), level, (tileWidth, tileHeight))
         tile.load()
         tile_image = PIL.Image.new("RGB", tile.size, (255, 255, 255))
         tile_image.paste(tile, mask=tile.split()[3])
         print("Image loaded: " + str(timer - time.time()))
         timer = time.time()

         tileHistogram= tile_image.histogram()
         if tileHistogram[255] == cutoff and tileHistogram[511] == cutoff and tileHistogram[767] == cutoff:
            print("Cutoff")
            #print(tileHistogram)
            continue
         if sum(tileHistogram[0:255]) == 0 and sum(tileHistogram[256:511]) == 0  and sum(tileHistogram[512:767]) == 0:
            print("Sum")
            #print(tileHistogram)
            continue
         print("Histogram Check: " + str(timer - time.time()))
         timer = time.time()

         tile_image.save(os.path.join(outputFolder, "x" + str(tilePositionX) + "_y" + str(tilePositionY) + '.jpg'))
         print("Save Image: " + str(timer - time.time()))
         timer = time.time()

         '''
         plt.figure(0)
         for i in range(0, 256):
            plt.bar(i, tileHistogram[0:256][i], color=getRed(i), edgecolor=getRed(i), alpha=0.3)
         plt.savefig(os.path.join(outputFolder, "x" + str(tilePositionX) + "_y" + str(tilePositionY) + '_r.jpg'))
         plt.figure(1)
         for i in range(0, 256):
            plt.bar(i, tileHistogram[256:512][i], color=getGreen(i), edgecolor=getGreen(i), alpha=0.3)
         plt.savefig(os.path.join(outputFolder, "x" + str(tilePositionX) + "_y" + str(tilePositionY) + '_g.jpg'))
         plt.figure(2)
         for i in range(0, 256):
            plt.bar(i, tileHistogram[512:768][i], color=getBlue(i), edgecolor=getBlue(i), alpha=0.3)
         plt.savefig(os.path.join(outputFolder, "x" + str(tilePositionX) + "_y" + str(tilePositionY) + '_b.jpg'))
         print("Save Histogram: " + str(timer - time.time()))
         timer = time.time()
         plt.close()
         '''

         tile_image.close()
         print("Run Closed: " + str(startFor - time.time()))

   slide.close()

   endTime = time.time()
   print("Slide " + str(imagename) + " [" + str(slideWidth) + "x" + str(slideHeight) + "] "+
         " tiled into " + str(tilesX * tilesY) + " Tiles (" + str(tilesX) + "x" + str(tilesY) + ") " +
         "in " + str(endTime - startTime) + " secounds.")