#!/usr/bin/env python

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from PIL.Image import Image

from imgstitch import __version__, lib

_PathLike = str | os.PathLike[str]

logger = logging.getLogger(__name__)


def _open_images(paths: Iterable[_PathLike]) -> list[Image]:
    images = []
    ok = False
    try:
        for path in paths:
            logger.info('Reading %s', path)
            image = PIL.Image.open(Path(path))
            images.append(image)
            if image.width * image.height == 0:
                raise lib.ImgStitchError(f'Empty image {image.size}: {path}')
        ok = True
        return images
    finally:
        # On error, close images already loaded
        if not ok:
            for image in images:
                image.close()


def _save_image(image: Image, output_path: _PathLike | None) -> None:
    if output_path is not None:
        logger.info('Writing %s', output_path)
        image.save(Path(output_path))
    else:
        image.save(sys.stdout.buffer, 'png')


def _path_or_stdout_arg(s: str) -> Path | None:
    return None if s == '-' else Path(s)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@', add_help=False)
    ap.add_argument('images', metavar='IMAGE', nargs=2, type=Path, help=argparse.SUPPRESS)
    ap.add_argument('images2', metavar='IMAGE', nargs='*', type=Path, help='Two or more image files, top to bottom')
    ap.add_argument('-h', '--help', action='help', help='Print this help message and exit')
    ap.add_argument('-V', '--version', action='version', version=__version__,
                    help='Print the version number and exit')
    ap.add_argument('-o', '--output', metavar='FILE', default='out.png', type=_path_or_stdout_arg,
                    help="Stitched output file, or '-' for standard output (default: %(default)s)")
    ap.add_argument('-v', '--verbose', action='count', default=0, help='Print messages about progress')
    ap.add_argument('--crop-header-height', metavar='HEIGHT', default=0, type=int,
                    help="Exclude first HEIGHT pixels from each image (default: %(default)s)")
    ap.add_argument('--crop-first', action=argparse.BooleanOptionalAction, default=True,
                    help="Crop/don't crop header from first image when using --crop-header-height")
    args = ap.parse_args()
    args.images += args.images2
    del args.images2
    return args


def main() -> None:
    args = _parse_args()

    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels))]
    # logging.basicConfig(level=log_level, format='%(message)s')
    logging.basicConfig(level=log_level, format='%(asctime)s %(message)s')
    logging.captureWarnings(True)

    try:
        # Image data is lazily loaded by Pillow
        images = _open_images(args.images)
        out = lib.stitch(*images, crop_header_height=args.crop_header_height, crop_first=args.crop_first)
        _save_image(out, args.output)
    except lib.ImgStitchError as e:
        logger.error('Error: %s', ', '.join(str(v) for v in e.args))
        sys.exit(e.exit_code)
    except FileNotFoundError as e:
        logger.error('%s: %s', e.strerror, e.filename)
        sys.exit(2)


if __name__ == '__main__':
    main()
