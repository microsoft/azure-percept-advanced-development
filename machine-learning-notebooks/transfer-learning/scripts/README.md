# Scripts

This directory contains some scripts that are useful for developing custom models for the Percept DK.
Each script is documented in the source code itself and should have a `--help` option, but we have provided some context
surrounding them in this README as well.

These scripts are utilized as part of the [transfer learning notebook](../transfer-learning-using-ssd.ipynb).
In this notebook, the scripts are used to create a subset of the COCO dataset that only contains images of bowls.
The subset is then converted from COCO to VOC formatting, which is compatible with the model training code in the notebook.

## convert_coco_to_voc.py

This script converts a [COCO](https://cocodataset.org/#home)-style computer vision dataset to a
[Pascal VOC](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)-style dataset. Apart from the directory layout differences,
the main difference between the two types of dataset is that COCO uses a single JSON file for each split of the dataset,
while VOC uses one XML file per image in a dataset.

Typical usage of this script looks like this:

```
python convert_coco_to_voc.py <path/to/coco> --target <name-of-output-directory>
```

This script can also exclude any categories you want (treat them as background), by means of the `--negative-categories` argument.

**NOTE** This script currently only handles bounding boxes. Segmentation annotations are ignored.

## create_coco_subset.py

This script should be used in conjunction with `filter_coco.py` to create a filtered COCO-style dataset. Specifically,
this script takes a COCO JSON file that has been filtered by `filter_coco.py` and copies only the images that contain the
categories of interest, creating a new dataset based on the filtered JSON file.

For example, if you use `filter_coco.py` to filter out everything but zebras from the original COCO dataset's JSON file,
the original COCO dataset will still contain images of people, cars, bowls, etc. In order to filter the dataset
based on the new JSON, you use this script, like so:

```
python create_coco_subset.py -i <path/to/coco/subset_JSON.json> -g <path/to/coco/images/directory> -o coco_filtered
```

## filter_coco.py

This script is taken from [here](https://github.com/immersive-limit/coco-manager/blob/master/filter.py)
under MIT license.

Its purpose is to take in a COCO JSON file and filter it down to a smaller JSON file that contains annotations
only for images that contain classes of interest.

An example use of this script is:

```
python filter_coco.py -i <path/to/coco_JSON.json> -o <name-of-filtered-json-file> -c [list of classes to include]
```
