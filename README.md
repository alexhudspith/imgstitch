# imgstitch

[![Tests](https://github.com/alexhudspith/imgstitch/actions/workflows/tests.yml/badge.svg)](https://github.com/alexhudspith/imgstitch/actions/workflows/tests.yml)

## Description

An automatic image alignment and stitching tool for images that overlap vertically. Its main purpose is to
stitch together screenshots with overlapping regions that are near pixel-perfect.

## Usage

    imgstitch [-h] [-V] [-o FILE] [-v] [--crop-header-height HEIGHT] [--crop-first | --no-crop-first] [IMAGE ...]

## Options

    -h, --help                     Print this help message and exit
    -V, --version                  Print the version number and exit
    --show                         Show the output file using the default application
    -o FILE, --output FILE         Stitched output file, or '-' for standard output (default: out.png)
    -v, --verbose                  Print messages about progress
    --crop-header-height HEIGHT    Exclude first HEIGHT pixels from each image (default: 0)
    --crop-first, --no-crop-first  Crop (default) or don't crop header from first image when using --crop-header-height

## Known Limitations

- Since horizontal overlaps are not considered, the images being stitched must all have equal width
- The alpha channel of images, if any, is ignored
