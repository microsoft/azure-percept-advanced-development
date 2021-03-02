"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Update Custom Vision Service's OpenVino model to use BGR[0-255] input.
"""
import argparse
import pathlib
import xml.etree.ElementTree as ET
import numpy as np


def find_mean_value(layers):
    try:
        return next(layer for layer in layers if layer.get('type') == 'Const' and layer.find('data').get('shape') == '1,3,1,1')
    except StopIteration:  # Some models don't have mean subtraction layer.
        return None


def find_first_conv_weights(layers, edges):
    first_conv = next(layer for layer in layers if layer.get('type') == 'Convolution')
    edge = next(e for e in edges if e.get('to-layer') == first_conv.get('id') and e.get('to-port') == '1')
    return next(layer for layer in layers if layer.get('id') == edge.get('from-layer'))


def read_np_array(data, layer):
    data_node = layer.find('data')
    offset = int(data_node.get('offset'))
    shape = [int(s) for s in data_node.get('shape').split(',')]
    count = np.product(shape)
    return np.frombuffer(data, np.float32, count=count, offset=offset).reshape(shape)


def write_np_array(data, layer, value):
    offset = int(layer.find('data').get('offset'))
    data[offset:offset+value.nbytes] = value.tobytes()


def update_cvs_openvino(xml_filepath, bin_filepath, out_bin_filepath):
    tree = ET.parse(xml_filepath)
    layers = tree.getroot().find('layers').findall('layer')
    edges = tree.getroot().find('edges').findall('edge')
    data = bytearray(bin_filepath.read_bytes())

    # Update the mean_value. Swap channels and multiply the values by 255.
    mean_value_layer = find_mean_value(layers)
    if mean_value_layer:
        mean_value = read_np_array(data, mean_value_layer)
        mean_value = mean_value[:, (2, 1, 0), :, :] * 255
        write_np_array(data, mean_value_layer, mean_value)

    # Update the first conv layer. Swap channels and divide the values by 255.
    weights_layer = find_first_conv_weights(layers, edges)
    weights = read_np_array(data, weights_layer)
    weights = weights[:, (2, 1, 0), :, :] / 255
    write_np_array(data, weights_layer, weights)

    out_bin_filepath.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description="Change OpenVIno input format from RGB[0-1] to BGR[0-255]")
    parser.add_argument('xml_filepath', type=pathlib.Path)
    parser.add_argument('bin_filepath', type=pathlib.Path)
    parser.add_argument('out_bin_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if not args.xml_filepath.exists() or not args.bin_filepath.exists():
        parser.error("Input files are not found.")

    if args.out_bin_filepath.exists():
        parser.error(f"{args.out_bin_filepath} already exists.")

    update_cvs_openvino(args.xml_filepath, args.bin_filepath, args.out_bin_filepath)


if __name__ == '__main__':
    main()
