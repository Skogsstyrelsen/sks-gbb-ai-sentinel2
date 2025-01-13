import argparse
import torch
import numpy as np
import pathlib
import platform
import os
from osgeo import gdal
import json
import pdb

from utils.model.load_model import load_model


pltf = platform.system()
if pltf == 'Linux':
    print("on Linux system")
    pathlib.WindowsPath = pathlib.PosixPath
else:
    print("on Windows system")
    pathlib.PosixPath = pathlib.WindowsPath



def plot_pred(pred, image_name, in_ds, crop_ammount):
    pred = np.clip(pred, 0, 1)

    write_geotiff(image_name,pred,in_ds, crop_ammount)

def read_geotiff(filename):
    ds = gdal.Open(filename)
    bands = ds.RasterCount
    xshape = ds.RasterXSize
    yshape = ds.RasterYSize
    nparray = np.zeros((bands,xshape,yshape), dtype='float32')
    for band in range( bands):
        nparray[band,...] = ds.GetRasterBand(band+1).ReadAsArray()
    return nparray, ds

def write_geotiff(filename, arr, in_ds, crop_amount):
    arr_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)

 
    gt = list(in_ds.GetGeoTransform())
    gt[0] += gt[1] * crop_amount  # adjust top left x
    gt[3] += gt[5] * crop_amount  # adjust top left y

    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(gt)

    arr = arr.squeeze(2)
    out_ds.GetRasterBand(1).WriteArray(arr)
    out_ds.FlushCache()
    out_ds = None  


def main(opt):
    print("main")
    selected_apis = []
    if opt.configuration != '':
        print("using config {}".format(opt.configuration))
        with open(opt.config_file,encoding='utf-8') as f:
            json_data = json.load(f)

        for api in json_data[opt.configuration]['apis']:
            print("api: {}".format(api))
            selected_apis.append(api)

    
    model, device = load_model(opt)

    inference_padding = opt.inference_padding

    image_name = opt.filename

    if not os.path.exists(image_name):
        print("name of file: {}".format(image_name))
        return
    arr, ds = read_geotiff(image_name)
    geoTransform = ds.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    miny = maxy + geoTransform[5] * ds.RasterYSize
    maxx = minx + geoTransform[1] * ds.RasterXSize
    
    
    # Calculate image coordinates in sweref99
    image_coordinates=[int(minx), int(miny), int(maxx), int(maxy)]

    image = arr/255.0
    image = torch.from_numpy(image).to(device)

    image = image[None, ...]
    pred = model(image.float())
    
    pred = pred.cpu().detach().numpy()[0].transpose(1,2,0)


    string_coordinates = ','.join([str(int(coord)) for coord in image_coordinates])
    image_base_name = '{}_[{}]'.format(opt.outfile, string_coordinates)


    plot_pred(pred,'{}/{}.tiff'.format(opt.output_path,image_base_name),ds, inference_padding)

def ensure_dirs(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Directory '{dir}' created successfully!")

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config')
    parser.add_argument('--inference_padding', type=int, default=0, help='Use to skip detections with zero values from input')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--input_path', type=str, default='input')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--filename', type=str, default='input.tiff', help='*.tiff')
    parser.add_argument('--outfile', type=str, default='output', help='*.gpkg')
    parser.add_argument('--input_channels', type=int, nargs='+', action='store', default=[3,3])
    parser.add_argument('--output_channels', type=int, action='store', default=1)
    parser.add_argument('--weights', type=str, default='model.pth', help='model.pth path(s)')
    parser.add_argument('--device', action='store', type=str, default='0')
    parser.add_argument('--algorithm', action='store', type=str, default='attentionpixelclassifier')
    parser.add_argument('--configuration', action='store', default='')
    parser.add_argument('--create_annotations', action='store_true')
    opt = parser.parse_args()
    
    ensure_dirs([opt.input_path,opt.output_path])
    main(opt)
