import logging
import time
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import signal, linalg

import PIL.Image
import PIL.ImageOps
from PIL.Image import Image

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


def _imshow(*arrays):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig, ax, = plt.subplots(1, len(arrays), sharex=True)
    if not hasattr(ax, '__len__'):
        ax = [ax]
    mn = min(a.min() for a in arrays)
    mx = max(a.max() for a in arrays)
    plt.ion()
    for i, a in enumerate(arrays):
        ax[i].imshow(a, vmin=mn, vmax=mx, interpolation='nearest', aspect='auto')


def _plot(v):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    fig, ax, = plt.subplots(1, 1)
    plt.ion()
    ax.plot(v)


def _allclose(a, b):
    return np.allclose(a, b, atol=2.0)


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
        logger.debug('- best #%d, a_row = %d, b_row = %d, correlation MSE: %f', i, a_row, b_row, corr[c])
        if a_row >= 0 and b_row < b.shape[0] and _allclose(a[a_row:], b[:b_row]):
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
