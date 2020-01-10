import os
from preprocessing.tile_image import TileSlide
from queue import Queue
from threading import Thread
from time import time

def preprocessing():
   ts = time()
   slideextension = "mrxs"
   #datasetpath = "G:\HEROHE_CHALLENGE\DataSet\HEROHE_CHALLENGE"
   #outputpath = "H:\HEROHE_CHALLENGE\DataSet\\tiled"
   datasetpath = "D:\GoogleDrive\Arbeit\HEROHE_Challenge\TEST"
   outputpath = "C:\work\HEROHE\\test\\tiled"

   # Preprocessing:
   queue = Queue()

   for x in range(16):
           worker = TileSlide(queue)
           # Setting daemon to True will let the main thread exit even though the workers are blocking
           worker.daemon = True
           worker.start()

   for root, dirs, files in os.walk(datasetpath, topdown=False):
      for name in files:
         if name.endswith(slideextension):
            queue.put((root, outputpath, name, 0))
            #print(os.path.join(root, name))
            #tileSlide( root, outputpath, name, 0)

   queue.join()
   print('Preprocessing took %s', time() - ts)

if __name__ == '__main__':
    preprocessing()
