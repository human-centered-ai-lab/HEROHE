import os
import time
import PIL
import json
import openslide
import matplotlib.pyplot as plt
from threading import Thread

class TileSlide(Thread):

   tileSizeX = 512
   tileSizeY = 512
   cutoff = tileSizeX*tileSizeY
   thumbnaillevel = 5

   def __init__(self, queue):
      Thread.__init__(self)
      self.queue = queue


   def run(self):
      while True:
         # Get the work from the queue and expand the tuple
         inputpath, outputpath, imagename, level = self.queue.get()
         try:
            self.tileSlide(inputpath, outputpath, imagename, level)
         finally:
            self.queue.task_done()

   def getRed(self, redVal):
      return '#%02x%02x%02x' % (redVal, 0, 0)


   def getGreen(self, greenVal):
      return '#%02x%02x%02x' % (0, greenVal, 0)


   def getBlue(self, blueVal):
      return '#%02x%02x%02x' % (0, 0, blueVal)

   def tileSlide(self, inputpath, outputpath, imagename, level):
      startTime = time.time()
      # Load Slide
      slide = openslide.OpenSlide(os.path.join(inputpath, imagename))

      # Create Folder
      fileExtenstionPosition = imagename.rfind(".")
      outputFolder = os.path.join(outputpath, str(imagename[:fileExtenstionPosition]) + "_" + str(self.tileSizeX) + "x" + str(self.tileSizeY))
      if not os.path.exists(outputFolder):
         os.makedirs(outputFolder)
      else:
         return

      # Create Json for Slide
      json_slide = {}
      json_slide["slide"] = imagename[:fileExtenstionPosition]
      json_slide["level_count"] = slide.level_count
      json_slide["dimensions"] = slide.dimensions
      json_slide["level_dimensions"] = slide.level_dimensions
      json_slide["level_downsamples"] = slide.level_downsamples
      #json_slide["properties"] = slide.properties
      #json_slide["associated_images"] = slide.associated_images

      # Create Thumbnail & H istogram
      thumbnail = slide.get_thumbnail(slide.level_dimensions[self.thumbnaillevel])
      thumbnail.load()
      thumbnail_histogram = thumbnail.histogram()
      thumbnail.save(os.path.join(outputFolder, str(imagename[:fileExtenstionPosition]) + "_thumbnail.jpg"))
      plt.figure(0)
      for i in range(0, 256):
         plt.bar(i, thumbnail_histogram[0:256][i], color=self.getRed(i), edgecolor=self.getRed(i), alpha=0.3)
      plt.savefig(os.path.join(outputFolder, str(imagename[:fileExtenstionPosition]) + "_thumbnail_histogram_r.jpg"))
      plt.figure(1)
      for i in range(0, 256):
         plt.bar(i, thumbnail_histogram[256:512][i], color=self.getGreen(i), edgecolor=self.getGreen(i), alpha=0.3)
      plt.savefig(os.path.join(outputFolder, str(imagename[:fileExtenstionPosition]) + "_thumbnail_histogram_g.jpg"))
      plt.figure(2)
      for i in range(0, 256):
         plt.bar(i, thumbnail_histogram[512:768][i], color=self.getBlue(i), edgecolor=self.getBlue(i), alpha=0.3)
      plt.savefig(os.path.join(outputFolder, str(imagename[:fileExtenstionPosition]) + "_thumbnail_histogram_b.jpg"))
      plt.close()
      json_slide["thumbnail_dimensions"] = slide.level_dimensions[self.thumbnaillevel]
      json_slide["thumbnail_histogram_r"] = thumbnail_histogram[0:256]
      json_slide["thumbnail_histogram_g"] = thumbnail_histogram[256:512]
      json_slide["thumbnail_histogram_b"] = thumbnail_histogram[512:768]


      slideWidth, slideHeight = slide.level_dimensions[level ]

      tilesX = int(slideWidth / self.tileSizeX) + 1
      tilesY = int(slideHeight / self.tileSizeY) + 1

      json_slide["tile"] = []

      for tilePositionX in range(tilesX):
         for tilePositionY in range(tilesY):
            json_tile = {}
            json_tile["tile"] = "x" + str(tilePositionX) + "_y" + str(tilePositionY)
            json_tile["pos_x"] = tilePositionX
            json_tile["pos_y"] = tilePositionY
            locationX = tilePositionX * self.tileSizeX
            locationY = tilePositionY * self.tileSizeY
            tileWidth = min(slideWidth, locationX + self.tileSizeX) - locationX
            tileHeight = min(slideHeight, locationY + self.tileSizeY) - locationY
            json_tile["tileWidth"] = tileWidth
            json_tile["tileHeight"] = tileHeight
            tile = slide.read_region((locationX, locationY), level, (tileWidth, tileHeight))
            tile.load()
            tile_image = PIL.Image.new("RGB", tile.size, (255, 255, 255))
            tile_image.paste(tile, mask=tile.split()[3])
            #print("Image loaded: " + str(timer - time.time()))
            #timer = time.time()


            tileHistogram= tile_image.histogram()
            if tileHistogram[255] == self.cutoff and tileHistogram[511] == self.cutoff and tileHistogram[767] == self.cutoff:
               #print("Cutoff")
               #print(tileHistogram)
               continue
            if sum(tileHistogram[0:255]) == 0 and sum(tileHistogram[256:511]) == 0  and sum(tileHistogram[512:767]) == 0:
               #print("Sum")
               #print(tileHistogram)
               continue
            json_tile["tile_histogram_r"] = tileHistogram[0:256]
            json_tile["tile_histogram_g"] = tileHistogram[256:512]
            json_tile["tile_histogram_b"] = tileHistogram[512:768]

            json_slide["tile"].append(json_tile)

            tile_image.save(os.path.join(outputFolder, "x" + str(tilePositionX) + "_y" + str(tilePositionY) + '.jpg'))
            #print("Save Image: " + str(timer - time.time()))
            #timer = time.time()

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
            #print("Run Closed: " + str(startFor - time.time()))

      slide.close()

      with open(os.path.join(outputFolder, str(imagename[:fileExtenstionPosition]) + "_data.json"), 'w') as outfile:
         json.dump(json_slide, outfile)

      endTime = time.time()
      print("Slide " + str(imagename) + " [" + str(slideWidth) + "x" + str(slideHeight) + "] "+
            " tiled into " + str(tilesX * tilesY) + " Tiles (" + str(tilesX) + "x" + str(tilesY) + ") " +
            "in " + str(endTime - startTime) + " secounds.")