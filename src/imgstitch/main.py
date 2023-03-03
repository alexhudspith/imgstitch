#!/usr/bin/env python

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from PIL.Image import Image

from imgstitch import __version__, lib

_PathLike = str | os.PathLike[str]

logger = logging.getLogger(__name__)

# Image suffixes considered when reading from directory
_IMAGE_SUFFIXES = {'.png', '.gif', '.jpeg', '.jpg'}

# os.startfile is only available on Windows
# Used for option --show
if hasattr(os, 'startfile'):
    _startfile = os.startfile
else:
    def _startfile(path):
        _cmd = 'xdg-open' if sys.platform.startswith('linux') else 'open'
        subprocess.Popen([_cmd, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


def _find_images(paths: list[Path], exclude: Path | None = None) -> list[Path]:
    result = []
    for p in paths:
        try:
            file_paths = [
                f for f in p.iterdir()
                if f.suffix in _IMAGE_SUFFIXES and not _samefile(f, exclude)
            ]
            result.extend(sorted(file_paths))
        except NotADirectoryError:
            result.append(p)

    return result


def _samefile(path1: Path | None, path2: Path | None):
    if path1 is None or path2 is None:
        return False

    try:
        return path1.samefile(path2)
    except FileNotFoundError:
        return False


def _path_or_stdout_arg(s: str) -> Path | None:
    return None if s == '-' else Path(s)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@', add_help=False, epilog="""
        For any directory IMAGE argument, the contained images are used in order of filename.
        The output file path is excluded, if it would otherwise be matched. Subdirectories are not searched.
        """)
    ap.add_argument('images', metavar='IMAGE', nargs='+', type=Path,
                    help='Image files and/or directories, ordered top to bottom')
    ap.add_argument('-h', '--help', action='help', help='Print this help message and exit')
    ap.add_argument('-V', '--version', action='version', version=__version__,
                    help='Print the version number and exit')
    ap.add_argument('--show', action='store_true',
                    help='Show the output file using the default application')
    ap.add_argument('-o', '--output', metavar='FILE', default='out.png', type=_path_or_stdout_arg,
                    help="Stitched output file, or '-' for standard output (default: %(default)s)")
    ap.add_argument('-v', '--verbose', action='count', default=0, help='Print messages about progress')
    ap.add_argument('--crop-header-height', metavar='HEIGHT', default=0, type=int,
                    help="Exclude first HEIGHT pixels from each image (default: %(default)s)")
    ap.add_argument('--crop-first', action=argparse.BooleanOptionalAction, default=True,
                    help="Crop (default) or don't crop header from first image when using --crop-header-height")

    args = ap.parse_args()
    args.images = _find_images(args.images, args.output)
    if len(args.images) < 2:
        ap.error(f'At least two images are required but have {len(args.images)}')

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
        if args.show:
            _startfile(args.output)
    except lib.ImgStitchError as e:
        logger.error('Error: %s', ', '.join(str(v) for v in e.args))
        sys.exit(e.exit_code)
    except FileNotFoundError as e:
        logger.error('%s: %s', e.strerror, e.filename)
        sys.exit(2)


if __name__ == '__main__':
    main()
