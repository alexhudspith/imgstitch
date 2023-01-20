#!/usr/bin/env python

import argparse
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from pathlib import Path

import numpy as np
from scipy import signal, linalg

import PIL.Image
import PIL.ImageOps
from PIL.Image import Image

from imgstitch import __version__


_PathLike = str | PathLike[str]

logger = logging.getLogger(__name__)


class ImgStitchError(RuntimeError):
    """Exception raised when stitching could not be completed."""

    def __init__(self, message: str | None = None, exit_code: int = 2):
        args = () if message is None else (message,)
        super().__init__(*args)
        self.exit_code = exit_code


def _image_name(image: Image, counter: int) -> str:
    r"""
    Return the image filename if available, otherwise 'Image #\ *counter*'.

    :param image: the image to name
    :param counter: integer suffix to use when no filename is available
    :return: the name
    """
    filename = image.filename if hasattr(image, 'filename') else None
    return filename or f'Image #{counter}'


def stitch(*images: Image, crop_header_height: int = 0, crop_first: bool = True) -> Image:
    """
    Stitch two or more overlapping images given in top-to-bottom order.

    :param images: vertically overlapping images to stitch together
    :param crop_header_height: rows to crop from top of each image
    :param crop_first: whether to crop the first image
    :return: the stitched image
    :raises ImgStitchError: if the images are not all of equal width,
        or if no overlap could be found
    """
    if len(images) < 2:
        raise ValueError(f'At least 2 image paths are required, got {len(images)}')

    if crop_header_height < 0:
        raise ValueError(f'Crop header height must be >= 0, got {crop_header_height}')

    _check_images_compatible(images)

    img_iter = iter(images)
    img_a = next(img_iter)

    if crop_header_height and crop_first:
        img_a = img_a.crop((0, crop_header_height) + img_a.size)

    for counter, img_b in enumerate(img_iter, 2):
        if img_a.mode != img_b.mode:
            name = _image_name(img_b, counter)
            logger.warning(f'Converting from {img_b.mode} to {img_a.mode}: {name}')
            img_b = img_b.convert(img_a.mode)

        if crop_header_height:
            img_b = img_b.crop((0, crop_header_height) + img_b.size)
        stitched = _stitch_two(img_a, img_b)
        img_a = stitched

    return stitched


def _stitch_two(img_a: Image, img_b: Image) -> Image:
    assert img_a.mode == img_b.mode
    assert img_a.width == img_b.width

    t0 = time.perf_counter_ns()

    logger.info('Stitching images')
    a = _image_to_array(img_a)
    b = _image_to_array(img_b)
    _normalize(a, b)

    a_row = None
    overlap = min(a.shape[0], b.shape[0])
    while overlap > 0:
        logger.debug('Trying overlap %d', overlap)
        a_row = _find_break_row(a, b, overlap)
        if a_row is not None:
            break
        overlap //= 2

    if a_row is None:
        raise ImgStitchError('No vertical overlap found between images')

    ab_size = img_a.width, a_row + img_b.height
    out = PIL.ImageOps.pad(img_a, ab_size, centering=(0, 0))
    out.paste(img_b, (0, a_row))

    t1 = time.perf_counter_ns()
    logger.info(f'Stitching took {(t1 - t0) * 1e-9:.2f}s')

    return out


def _check_images_compatible(images: Iterable[Image]) -> None:
    images_by_width = defaultdict(list)
    for image in images:
        images_by_width[image.width].append(image)

    if len(images_by_width) != 1:
        for counter, (width, image_list) in enumerate(images_by_width.items(), 1):
            logger.error(f'({width} x _):')
            for image in image_list:
                logger.error(f'  {_image_name(image, counter)}')
        raise ImgStitchError(f'Images have different widths: {sorted(images_by_width.keys())}')


def _plot(a, b):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig, ax, = plt.subplots(1, 2, sharex=True, sharey=True)
    mn = min(a.min(), b.min())
    mx = max(a.max(), b.max())
    ax[0].imshow(a, vmin=mn, vmax=mx, interpolation='nearest', aspect='auto')
    ax[1].imshow(b, vmin=mn, vmax=mx, interpolation='nearest', aspect='auto')
    plt.show()


def _find_break_row(a: np.ndarray, b: np.ndarray, overlap: int, try_best_n: int = 4) -> int | None:
    """
    Find the earliest row of ``a`` at which ``b`` starts overlapping.

    :param a: top 2D array
    :param b: bottom 2D array
    :param overlap: maximum number of overlap rows to consider
    :param try_best_n: number of highest ranked correlations to try
    :return: first row index of ``a`` where all remaining rows match ``b``
    """
    a_offset = len(a) - overlap
    a_overlap = a[-overlap:]
    b_overlap = b[:overlap]
    corr = _norm_cross_correlation(a_overlap, b_overlap)
    corr_mse = np.sum((corr - 1) ** 2, axis=1)
    best_corr = np.argsort(corr_mse)[:try_best_n]
    # max_row2 = _arg_last(np.sum(corr, axis=1), np.argmax)
    for i, c in enumerate(best_corr, 1):
        a_row = c - len(b_overlap) + 1 + a_offset
        b_row = a.shape[0] - a_row
        logger.debug('- best #%d, a_row = %d, b_row = %d, correlation MSE: %f', i, a_row, b_row, corr_mse[c])
        if a_row >= 0 and b_row < b.shape[0] and np.allclose(a[a_row:], b[:b_row]):
            break
    else:
        a_row = None
    return a_row


def _normalize(a: np.ndarray, b: np.ndarray, axis: int = 0) -> None:
    """Remove offset/bias from arrays."""
    a_means = a.mean(axis=axis)
    a -= a_means
    b -= a_means


def _norm_cross_correlation(a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the normalized cross-correlation along one axis."""
    c: np.ndarray = signal.fftconvolve(a, b[::-1], mode='full', axes=axis)
    norm = linalg.norm(a, axis=axis) * linalg.norm(b, axis=axis)
    norm[np.isclose(norm, 0)] = 1
    c /= norm
    return c


def _image_to_array(image: Image) -> np.ndarray:
    """Return the given PIL image as a NumPy 2D float array."""
    a = np.array(image, dtype=np.float64, order='C')
    if a.ndim > 2:
        n = a.shape[2]
        a = np.hstack([a[:, :, i] for i in range(min(n, 3))])
    return a


def _open_images(paths: Iterable[_PathLike]) -> list[Image]:
    images = []
    ok = False
    try:
        for path in paths:
            logger.info('Reading %s', path)
            image = PIL.Image.open(Path(path))
            images.append(image)
            if image.width * image.height == 0:
                raise ImgStitchError(f'Empty image {image.size}: {path}')
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
        out = stitch(*images, crop_header_height=args.crop_header_height, crop_first=args.crop_first)
        _save_image(out, args.output)
    except ImgStitchError as e:
        logger.error('Error: %s', ', '.join(str(v) for v in e.args))
        sys.exit(e.exit_code)
    except FileNotFoundError as e:
        logger.error('%s: %s', e.strerror, e.filename)
        sys.exit(2)


if __name__ == '__main__':
    main()
