import logging
import time
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import signal

import PIL.Image
import PIL.ImageOps
import PIL.ImageChops
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


def _find_variation(b, axis=1) -> int:
    non_zero = np.any(np.abs(b) > 1e-10, axis=axis)
    nz = np.flatnonzero(non_zero)
    return nz[0] if len(nz) != 0 else 0


def _image_overlap_equal(img_a, img_b, a_row):
    b_row = img_a.height - a_row
    a_bottom = img_a.crop((0, a_row, img_a.width, img_a.height)).convert('RGB')
    b_top = img_b.crop((0, 0, img_b.width, b_row)).convert('RGB')
    diff = PIL.ImageChops.difference(a_bottom, b_top)
    return diff.getbbox() is None


def _stitch_two(img_a: Image, img_b: Image) -> Image:
    assert img_a.mode == img_b.mode
    assert img_a.width == img_b.width

    t0 = time.perf_counter_ns()

    logger.info('Stitching images')
    a = _image_to_array(img_a)
    b = _image_to_array(img_b)
    _normalize(a, b)

    b_row = _find_variation(b)

    result_a_row = None
    max_overlap = min(a.shape[0], b.shape[0])
    filter_height = min(b_row + 1, max_overlap)

    while result_a_row is None and filter_height <= max_overlap:
        logger.debug('Trying filter height %d', filter_height)
        for a_row in _find_break_row(a, b, filter_height):
            if _image_overlap_equal(img_a, img_b, a_row):
                result_a_row = a_row
                break

        filter_height *= 2

    if result_a_row is None:
        raise ImgStitchError('No vertical overlap found between images')

    ab_size = img_a.width, result_a_row + img_b.height
    out = PIL.ImageOps.pad(img_a, ab_size, centering=(0, 0))
    out.paste(img_b, (0, result_a_row))

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


def _find_break_row(a: np.ndarray, b: np.ndarray, filter_height: int, min_try_corr_factor: float = 0.99) -> Iterable[int]:
    """
    Yield row indexes of ``a`` at which ``b`` might start overlapping.

    :param a: top 2D array
    :param b: bottom 2D array
    :param filter_height: rows of b to use in correlation
    :param min_try_corr_factor: try matching using correlations down to
        this proportion of the maximum correlation (0.0 < threshold ≤ 1.0)
    :return: first row index of ``a`` where could ``b``  overlap
    """
    max_overlap = min(a.shape[0], b.shape[0])
    a_offset = len(a) - max_overlap
    a_overlap = a[-max_overlap:]
    b_filter = b[:filter_height]
    corr = _cross_correlation(a_overlap, b_filter)[filter_height - 1:]

    # Correlations are in order of increasing rows of 'a'. Stable sort
    # to maintain this ordering among equal correlations so larger
    # overlaps are considered first.
    a_rows = np.argsort(-corr, kind='stable')
    max_corr = corr[a_rows[0]]

    result_a_row = None
    for a_row in a_rows:
        if corr[a_row] / max_corr < min_try_corr_factor:
            # Correlation too low, stop
            break
        yield a_row + a_offset

    return result_a_row


def _normalize(a: np.ndarray, b: np.ndarray, axis=1) -> None:
    """Remove offset/bias from arrays."""
    mean = np.mean(a, axis=axis, keepdims=True)
    a -= mean
    a_mean = np.mean(mean)

    mean = np.mean(b, axis=axis, keepdims=True)
    b -= mean

    stddev = np.sqrt(np.mean(a * a, axis=axis, keepdims=True))
    stddev[np.isclose(stddev, 0)] = 1
    a /= stddev

    stddev = np.sqrt(np.mean(b * b, axis=axis, keepdims=True))
    stddev[np.isclose(stddev, 0)] = 1
    b /= stddev

    if a_mean >= 128:
        a *= -1
        b *= -1


def _cross_correlation(a: np.ndarray, b: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the cross-correlation along one axis."""
    c: np.ndarray = signal.fftconvolve(a, b[::-1], mode='full', axes=axis)
    c = np.sum(c, axis=1)
    return c


def _image_to_array(image: Image) -> np.ndarray:
    """Return the given PIL image as a NumPy 2D float array."""
    a = np.array(image, dtype=np.float64, order='C')
    if a.ndim > 2:
        n = a.shape[2]
        a = np.hstack([a[:, :, i] for i in range(min(n, 3))])
    return a
