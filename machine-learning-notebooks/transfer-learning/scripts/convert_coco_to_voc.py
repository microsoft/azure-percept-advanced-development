"""
Converts a COCO-style dataset to a VOC-style dataset.

The difference between the two styles is that the annotation files are different.
VOC uses one XML file for each image. COCO uses one JSON for each split in the dataset.

Ignores any segmentation (we are only interested in bounding boxes for now).
"""
from tqdm import tqdm
import argparse
import json
import os
import shutil

# A template for XML files pertaining to images which contain no usable objects
NEGATIVE_XML_TEMPLATE = \
"""
<annotation>
    <folder>images</folder>
    <filename>$FILENAME$</filename>
    <path>$PATH$</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>$WIDTH$</width>
        <height>$HEIGHT$</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>negative</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
    </object>
</annotation>
""".lstrip()

# A template for XML files pertaining to images which contain one item of interest
POSITIVE_XML_TEMPLATE = \
"""
<annotation>
    <folder>images</folder>
    <filename>$FILENAME$</filename>
    <path>$PATH$</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>$WIDTH$</width>
        <height>$HEIGHT$</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
</annotation>
""".lstrip()

# A single object for adding more than one annotation to an image in XML
XML_OBJECT_TEMPLATE = \
"""
    <object>
        <name>$OBJECT_NAME$</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>$BNDBOX_XMIN$</xmin>
            <ymin>$BNDBOX_YMIN$</ymin>
            <xmax>$BNDBOX_XMAX$</xmax>
            <ymax>$BNDBOX_YMAX$</ymax>
        </bndbox>
    </object>
"""

def _convert_coco_bounding_box_to_voc(x: int, y: int, width: int, height: int) -> (int, int, int, int):
    """
    Converts the x, y, width, height style bounding box to xmin, ymin, xmax, ymax style.
    """
    xmin = int(x)
    xmax = int(xmin + width)
    ymin = int(y)
    ymax = int(y + height)

    return xmin, ymin, xmax, ymax

def _create_negative_xml(image: dict, basedpath: str, imagesdpath: str) -> (str, str):
    """
    Creates a single XML file's contents out of the given image dict. Also
    returns the path where the XML file should go.
    """
    fname = os.path.basename(image['file_name'])
    fname = "negative_" + os.path.splitext(fname)[0] + ".xml"
    fpath = os.path.join(basedpath, fname)
    imgfpath = os.path.join(imagesdpath, "negative_" + os.path.basename(image['file_name']))

    xmlsrc = NEGATIVE_XML_TEMPLATE.replace("$FILENAME$", os.path.basename(imgfpath))
    xmlsrc = xmlsrc.replace("$PATH$", imgfpath)
    xmlsrc = xmlsrc.replace("$WIDTH$", str(image['width']))
    xmlsrc = xmlsrc.replace("$HEIGHT$", str(image['height']))

    return xmlsrc, fpath

def _create_positive_xml(image: dict, basedpath: str, imagesdpath: str, annotations: [dict], negative_categories: {str}, categories: {int: str}) -> (str, str):
    """
    Creates a single XML file's contents out of the given image dict (and associated args).
    Also returns the path where the XML file should go.

    Args
    ----

    - image:                The image dict from the JSON file
    - basedpath:            The directory that this XML is going to go in
    - imagesdpath:          The directory where all the images will live.
    - annotations:          A list of annotations from the JSON file, corresponding to this image
    - negative_categories:  A dict of category names that we will ignore while adding annotations to the XML
    - categories:           A dict that maps category IDs to category names

    """
    # Determine which class of items is in this image. Find the first category not in the negative set.
    assert len(annotations) > 0, "Got 0 annotations for a positive XML. This XML should be negative. Original image file name: {}".format(image['file_name'])
    classname = None
    for annotation in annotations:
        category_name = categories[annotation['category_id']]
        if category_name not in negative_categories:
            classname = category_name
            break
    assert classname is not None

    fname = os.path.basename(image['file_name'])
    fname = classname + "_" + os.path.splitext(fname)[0] + ".xml"
    fpath = os.path.join(basedpath, fname)
    imgfpath = os.path.join(imagesdpath, classname + "_" + os.path.basename(image['file_name']))

    xmlsrc = POSITIVE_XML_TEMPLATE.replace("$FILENAME$", os.path.basename(imgfpath))
    xmlsrc = xmlsrc.replace("$PATH$", imgfpath)
    xmlsrc = xmlsrc.replace("$WIDTH$", str(image['width']))
    xmlsrc = xmlsrc.replace("$HEIGHT$", str(image['height']))

    for annotation in annotations:
        category_name = categories[annotation['category_id']]
        if category_name not in negative_categories:
            obj = XML_OBJECT_TEMPLATE.replace("$OBJECT_NAME$", category_name)
            xmin, ymin, xmax, ymax = _convert_coco_bounding_box_to_voc(*annotation['bbox'])
            obj = obj.replace("$BNDBOX_XMIN$", str(xmin))
            obj = obj.replace("$BNDBOX_YMIN$", str(ymin))
            obj = obj.replace("$BNDBOX_XMAX$", str(xmax))
            obj = obj.replace("$BNDBOX_YMAX$", str(ymax))

            # Splice in the object XML in the right place
            xmlsrc = xmlsrc.rstrip().rstrip("</annotation>").rstrip()
            xmlsrc += obj
            xmlsrc += "</annotation>"

    return xmlsrc, fpath


def convert_coco_json_to_voc_xml(jfpath: str, basedpath: str, negative_categories: set) -> ([str], [str]):
    """
    Converts each item in a COCO JSON file to one XML string each and returns the list of them
    along with the file path that it should be created at.

    Args
    ----

    - jfpath:       A path to a JSON file
    - basedpath:    The annotations directory of the VOC dataset.

    Returns
    -------

    A tuple of the form (list of XML file contents, list of XML file locations)

    """
    if negative_categories is None:
        negative_categories = set()

    # Load in the JSON file
    with open(jfpath) as f:
        json_contents = json.load(f)

    # Read out just the useful stuff
    images = json_contents['images']
    annotations = json_contents['annotations']
    categories = {category['id']: category['name'] for category in json_contents['categories']}
    imagesdpath = os.path.join(os.path.dirname(basedpath).split(os.sep)[0], "images")

    # Print some information
    print("Found {} images with annotations.".format(len(images)))
    print("Found {} total annotations.".format(len(annotations)))

    # Each image/annotation is one XML file in VOC style
    xmls = []
    xmlpaths = []
    negatives = 0
    positives = 0
    print("Creating XML annotation files for each image...")
    for image in tqdm(images):
        # Get all annotations corresponding to this image
        this_image_annotations = [annotation for annotation in annotations if annotation['image_id'] == image['id']]

        # If all the annotations in this image are from the negative list, let's create a negative XML
        negative = True
        for annotation in this_image_annotations:
            if categories[annotation['category_id']] not in negative_categories:
                negative = False
                break

        if negative:
            # We don't care about the annotations in this file. Create a negative instance out of it
            xml, xmlfpath = _create_negative_xml(image, basedpath, imagesdpath)
            negatives += 1
        else:
            # Normal case
            xml, xmlfpath = _create_positive_xml(image, basedpath, imagesdpath, this_image_annotations, negative_categories, categories)
            positives += 1
        xmls.append(xml)
        xmlpaths.append(xmlfpath)

    print("Total negative instances:", negatives)
    print("Total positive instances:", positives)

    return xmls, xmlpaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coco", type=str, help="Path to the root of the COCO-style dataset.")
    parser.add_argument("--negative-categories", '-n', type=str, nargs='+', default=None, help="A list of category names from the COCO dataset. When we encounter images with these annotations, we will treat them as negative.")
    parser.add_argument("--target", type=str, default="new-voc-dataset", help="Name of the new dataset.")
    args = parser.parse_args()

    # Sanity check the args
    if not os.path.isdir(args.coco):
        print("We need a path to the COCO-style dataset, but given {}".format(args.coco))
        exit(1)

    if os.path.isfile(args.target) or (os.path.isdir(args.target) and os.listdir(args.target)):
        print("Given --target of {}, but this already exists and is not empty.".format(args.target))
        exit(2)

    # Create the right directory structure for a VOC style dataset
    annotations_dpath = os.path.join(args.target, "annotations", "xmls")
    os.makedirs(args.target, exist_ok=True)
    os.makedirs(annotations_dpath)

    # Read in the COCO json files
    coco_annotations_dpath = os.path.join(args.coco, "annotations")
    json_fpaths = [os.path.join(coco_annotations_dpath, fpath) for fpath in os.listdir(coco_annotations_dpath) if os.path.splitext(fpath)[-1].lower() == ".json"]

    # Create an XML file for each item in each JSON file and put it in the right place
    all_xml_fpaths = []
    for jfpath in json_fpaths:
        print("Working on {}...".format(jfpath))
        xmls, xml_fpaths = convert_coco_json_to_voc_xml(jfpath, annotations_dpath, args.negative_categories)
        print("Writing XML files to disk...")
        for xml_contents, xfpath in tqdm(zip(xmls, xml_fpaths)):
            # Put the XML file in the right place
            with open(xfpath, 'w') as f:
                f.write(xml_contents + "\n")
            all_xml_fpaths.append(xfpath)

    # Copy the image files over
    images_dpath = os.path.join(args.target, "images")
    coco_images_dpath = os.path.join(args.coco, "images")
    print("Copying image files from {} to {}...".format(coco_images_dpath, images_dpath))
    shutil.copytree(coco_images_dpath, images_dpath)

    # Adjust each image file's name to match the class found in its corresponding XML
    print("Changing all image names to match their classes...")
    for xfpath in tqdm(all_xml_fpaths):
        xmlname = os.path.splitext(os.path.basename(xfpath))[0]
        assert "_" in xmlname, "File name does not contain any underscores. I don't know how to handle this: {}".format(xfpath)
        assert xmlname.split("_")[1] == xmlname.split("_")[-1], "File name contains more than one underscore. I don't know how to handle this: {}".format(xfpath)
        classname, imgfname = xmlname.split("_")
        original_img_fpath = os.path.join(images_dpath, imgfname + ".jpg")
        shutil.move(original_img_fpath, os.path.join(images_dpath, classname + "_" + imgfname + ".jpg"))
