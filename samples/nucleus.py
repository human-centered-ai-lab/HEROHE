"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
#from read_her_test import load_her2
import tqdm
import json
from deconvolution import Deconvolution
from skimage import img_as_float
import re

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) # To find local version of the library
from models.mrcnn.config import Config
from models.mrcnn import utils
from models.mrcnn import model as modellib
from models.mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs1")

VAL_IMAGE_IDS = [str(i) for i in range(20)]

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch

    STEPS_PER_EPOCH = 9175 // IMAGES_PER_GPU
    VALIDATION_STEPS = 400 // IMAGES_PER_GPU

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset, write_val=False):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "nucleus")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified json_file (Structure <image_id>:<image_path>)
        assert subset in ["train", "val", "test"]
        dataset_alt = "/home/simon/PycharmProjects/her2/Herohe2"
        slide_class_dirs = next(os.walk(dataset_alt))
        done = False
        if subset == "train":
            for i, item in enumerate(slide_class_dirs[1]):
                if ("neg" in item):
                    path_neg = os.path.join(dataset_alt, item)
                    # print(path_neg)
                    slides_neg = next(os.walk(path_neg))[1]
                    for path in slides_neg:

                        slide_path = os.path.join(path_neg, path)
                        slide_data_paths = next(os.walk(slide_path))[1]
                        slide_img_ids = []
                        # slide_img_ids = [(item.split("_")[0] + "_" + item.split("_")[1]) for item in slide_data_paths]
                        for item in slide_data_paths:
                            if "empty" in item:
                                continue
                            else:
                                slide_img_ids.append(item.split("_")[0] + "_" + item.split("_")[1])

                        for j, image_id in enumerate(slide_img_ids):
                            # print(os.path.join(slide_path, image_id + ".jpg"))
                            # input()
                            if write_val:
                                if (1 - np.random.uniform()) < 0.2:
                                    self.val_selection.update({image_id: os.path.join(slide_path, image_id + ".jpg")})
                                else:
                                    self.add_image(
                                        "nucleus",
                                        image_id=image_id,
                                        path=os.path.join(slide_path, image_id + ".jpg"))
                            else:
                                if not self.val_selection.get(image_id):
                                    self.add_image(
                                        "nucleus",
                                        image_id=image_id,
                                        path=os.path.join(slide_path, image_id + ".jpg"))
                            if j >= 1325:
                                break
                        # done = True
                        # if done:
                        #     break

                            # mask, mask_len = self.load_mask(0)
                            # print(mask.shape, mask_len)
                            # input()
                elif ("pos" in item):
                    path_neg = os.path.join(dataset_alt, item)
                    #print(path_neg)
                    slides_pos = next(os.walk(path_neg))[1]
                    for path in slides_pos:
                        #print(os.path.join(path_neg, path))
                        slide_path = os.path.join(path_neg, path)
                        slide_data_paths = next(os.walk(slide_path))[1]
                        slide_img_ids = []
                        for item in slide_data_paths:
                            if "empty" in item:
                                continue
                            else:
                                slide_img_ids.append(item.split("_")[0] + "_" + item.split("_")[1])

                        for image_id in slide_img_ids:

                            #print(os.path.join(slide_path, image_id + ".jpg"))
                            if write_val:
                                if (1 - np.random.uniform()) < 0.2:
                                    self.val_selection.update({image_id: os.path.join(slide_path, image_id + ".jpg")})
                                else:
                                    self.add_image(
                                        "nucleus",
                                        image_id=image_id,
                                        path=os.path.join(slide_path, image_id + ".jpg"))
                            else:
                                if not self.val_selection.get(image_id):
                                    self.add_image(
                                        "nucleus",
                                        image_id=image_id,
                                        path=os.path.join(slide_path, image_id + ".jpg"))
            if write_val:
                with open("val_files_her2.json", "w")as fp:
                    json.dump(self.val_selection, fp)

        elif subset == "val" or "test":
            for item in self.val_selection:
                self.add_image(
                    "nucleus",
                    image_id=item,
                    path = self.val_selection[item])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(info['path']), info["id"] + "_polygons_masks")

        # Read mask files from .jpg image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".jpg") and "total" not in f:
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)


        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset, valset):
    """Train the model."""
    # Training dataset.

    dataset_train = NucleusDataset()
    if valset is None:
        print("Warning rewriting Valset")
        val_dict = {}
        create_val = True
    else:
        with open(valset) as fp:
            val_dict = json.load(fp)
        create_val = False
        dataset_train.val_selection = val_dict


    dataset_train.load_nucleus(dataset_dir, subset, write_val=create_val)

    if valset is None:
        val_dict = dataset_train.val_selection

    print(len(dataset_train.image_info))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.val_selection = val_dict
    dataset_val.load_nucleus(dataset_dir, "val", create_val)
    print(len(dataset_val.image_info))
    input()
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=45,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                augmentation=augmentation,
                layers='all')


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, valdict):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Read dataset
    submit_dir = "/home/simon/PycharmProjects/her2/Mask_RCNN/samples/nucleus/Tests_fun3"
    file_id = valdict
    dataset = NucleusDataset()
    read_thresh = 0.65
    with open(file_id, "r") as fp:
        file_dict = json.load(fp)
    create_val = False
    dataset.val_selection = file_dict
    dataset.load_nucleus(dataset_dir, subset, create_val)
    dataset.prepare()

    #params:
    area_nuc = []
    area_nuc_dict = []
    roundness_slide = []
    roundness_instace = []
    area_bg = []
    image_ids_dict = []
    slide_id = None
    slide_stats = {}
    # Load over images
    j = 0
    l = 0

    for i, item in tqdm.tqdm(enumerate(dataset.image_info)):
        # Load image and run detection
        image = skimage.io.imread(item['path'])
        image_id = item['id']
        if i == 0:
            slide_id = item['path'].split("/")[-2]
            image_dir = os.path.join(submit_dir, slide_id)
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            slide_class = item['path'].split("/")[-3]
            slide_stats.update({slide_id: {'class': slide_class[0:3]}})
        elif slide_id != item['path'].split("/")[-2]:
            hist = np.histogram(area_nuc, bins="auto")
            num_dects = len(area_nuc)
            np.save(os.path.join(submit_dir, 'hist_data_{}_{}_{}'.format(slide_id,
                                                                         slide_stats[slide_id]['class'],
                                                                         read_thresh)), hist)
            slide_stats[slide_id].update({"area_nuc_arr":  str(area_nuc_dict),
                                          "area_nuc_total": str(np.sum(area_nuc)),
                                          "area_bg_arr": str(area_bg),
                                          "area_bg": str(np.sum(area_bg)),
                                          "roundness_instance": str(roundness_instace),
                                          "slide_roundness_arr": str(roundness_slide),
                                          "slide_round_mean": str(np.mean(roundness_slide)),
                                          "ids": str(image_ids_dict)})
            fig = plt.figure()
            plt.hist(area_nuc, bins="auto")
            plt.title("Mask area {}, n={}, cutoff={}".format(slide_id, num_dects, read_thresh))
            plt.savefig(os.path.join(submit_dir, 'hist_test_{}_{}.png'.format(slide_id, read_thresh)))

            area_nuc = []
            area_nuc_dict = []
            roundness_slide = []
            roundness_instace = []
            area_bg = []
            image_ids_dict = []

            slide_id = item['path'].split("/")[-2]
            image_dir = os.path.join(submit_dir, slide_id)
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            slide_class = item['path'].split("/")[-3]
            slide_stats.update({slide_id: {'class': slide_class[0:3]}})
            j = 0
        if j > 10:
            continue
        image_ids_dict.append(image_id)
        #use coler deconv VERY SLOW !!!

        #first_density, second_density, third_density = dec.out_scalars()
        # Produce reconstructed image, first layer, second layer, third layer and rest
        # out_images = dec.out_images(mode=[0, 1, 2, 3, -1])
        # image = np.array(out_images[2])

        # Detect objects
        r = model.detect([image], verbose=0)[0]
        area_nuc_instance = []
        for k, item in enumerate(r['rois']):
            if np.floor(np.sum(r['masks'][:, :, k])) == 0 or r['scores'][k] < read_thresh:
                continue
            else:
                area_nuc_instance.append(np.sum(r['masks'][:, :, k]))
                area_nuc.append(np.sum(r['masks'][:, :, k]))
        area_bg.append(image.shape[:2][0]**2 - np.sum(area_nuc_instance))

        area_nuc_dict.append(area_nuc_instance)

        # Save image with masks
        roundness, dect_image, roundness_nuc = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=True, read_thresh=read_thresh)

        roundness_slide.append(roundness)
        roundness_instace.append(roundness_nuc)
        dect_image = img_as_float(dect_image)/255
        plt.savefig("{}/{}/{}.png".format(submit_dir, slide_id, image_id.split("/")[-2] + '_' + image_id.split("/")[-1]))
        plt.imsave("{}/{}/{}_dects.png".format(submit_dir, slide_id, image_id.split("/")[-2] + '_' + image_id.split("/")[-1]),
                   dect_image)
        # plt.tight_layout()
        plt.close('all')

        if i == (len(dataset.image_info) - 1):
            hist = np.histogram(area_nuc, bins="auto")
            num_dects = len(area_nuc)
            np.save(os.path.join(submit_dir, 'hist_data_{}_{}_{}'.format(slide_id,
                                                                         slide_stats[slide_id]['class'],
                                                                         read_thresh)), hist)
            slide_stats[slide_id].update({"area_nuc_arr": str(area_nuc_dict),
                                          "area_nuc_total": str(np.sum(area_nuc)),
                                          "area_bg_arr": str(area_bg),
                                          "area_bg": str(np.sum(area_bg)),
                                          "roundness_instance": str(roundness_instace),
                                          "slide_roundness_arr": str(roundness_slide),
                                          "slide_round_mean": str(np.mean(roundness_slide)),
                                          "ids": str(image_ids_dict)})
            fig = plt.figure()
            plt.hist(area_nuc, bins="auto")
            plt.title("Mask area {}, n={}, cutoff={}".format(slide_id, num_dects, read_thresh))
            plt.savefig(os.path.join(submit_dir,'hist_test_{}_{}_{}.png'.format(slide_id, read_thresh,
                                                                                slide_stats[slide_id]['class'])))
        j+=1
     
    with open(os.path.join(submit_dir, "slide_stats_test_{}_pos.json".format(read_thresh)), "w") as fp:
        json.dump(slide_stats, fp)

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--valset', required=False,
                        metavar="Create a 0.2 portion of train data as val set",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset, args.valset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset, args.valset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
