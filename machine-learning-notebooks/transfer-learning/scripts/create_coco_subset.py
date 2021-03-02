"""
This script takes a JSON file that is a subset of the original
COCO JSON file and retrieves all the images.

The output should be a directory like this:

```
coco_subset/
    - annotations/
        - instances_train2017_filtered.json
        - instances_val2017_filtered.json
    images/
        - train2017_filtered/
            - ...
        - val2017_filtered/
            - ...
```
"""
from tqdm import tqdm
import argparse
import json
import os
import shutil


def _get_all_images_from_jsons(jfpath: str) -> [str]:
    """
    Gets the name of each image file from all the given JSON files.
    """
    ret = []
    for jfpath in args.input_jsons:
        with open(jfpath) as f:
            json_contents = json.load(f)

        # Read out just the useful stuff
        images = json_contents['images']

        for image in images:
            ret.append(image['file_name'])

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsons", '-i', nargs='+', type=str, help="List of paths to JSON files to use to compose the dataset (train, val, test, for example)")
    parser.add_argument("--input-imgs", '-g', nargs='+', type=str, help="List of paths to each directory containing images.")
    parser.add_argument("--output", '-o', type=str, default="coco_subset", help="The name of the directory we will create.")
    args = parser.parse_args()

    # Sanity check args
    for jsonfpath in args.input_jsons:
        if not os.path.exists(jsonfpath):
            print("{} does not exist.".format(jsonfpath))
            exit(1)

    for imgdpath in args.input_imgs:
        if not os.path.isdir(imgdpath):
            print("{} is not a directory. Need a path to each directory containing images.".format(imgdpath))
            exit(2)

    if os.path.exists(args.output):
        print("Output directory {} already exists.".format(args.output))
        exit(3)

    # Read in the JSON files to get all the image file names
    print("Reading in all images from the JSON files...")
    imgfnames = _get_all_images_from_jsons(args.input_jsons)

    # Set up the resulting dataset
    os.makedirs(args.output)
    os.makedirs(os.path.join(args.output, "annotations"))
    os.makedirs(os.path.join(args.output, "images"))

    # Copy over the JSON files
    for jfpath in args.input_jsons:
        jfname = os.path.basename(jfpath)
        shutil.copyfile(jfpath, os.path.join(args.output, "annotations", jfname))

    # Make a dict of all image names to their locations
    image_name_to_path = {}
    print("Caching all images...")
    for imgfname in tqdm(imgfnames):
        found = False
        for imagedpath in args.input_imgs:
            possible_fpath = os.path.join(imagedpath, imgfname)
            if os.path.isfile(possible_fpath):
                image_name_to_path[imgfname] = possible_fpath
                found = True
                break
        if not found:
            print("Could not find image: {} in any image directory. Excluding it.".format(imgfname))

    # Now copy all images over to the new location
    print("Copying images...")
    for imgfname in tqdm(imgfnames):
        shutil.copyfile(image_name_to_path[imgfname], os.path.join(args.output, "images", imgfname))
