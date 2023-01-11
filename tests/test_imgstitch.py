import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image
from imgstitch.main import stitch


# Set up test arguments
_IMAGE_FIXTURES = ('bird', 'amsterdam', 'mandelbrot', 'noise')
_PX = (5, 12, 100, 1000)
_ARG_NAMES = ('image_name', 'width', 'height', 'crop_y_1', 'crop_y_2')
_ARG_VALUES = sorted({
    (image, width, height, crop_y_1, crop_y_2)
    for image in _IMAGE_FIXTURES
    for width in _PX
    for height in _PX
    for crop_y_1 in (1, 2, height // 4, height // 2, 3 * height // 4, height - 1, height - 2)
    for crop_y_2 in (1, 2, height // 4, height // 2, 3 * height // 4, height - 1, height - 2)
    if crop_y_2 < crop_y_1
})
_IDS = [f'{i}, w={w}, h={h}, y1={y1}, y2={y2}' for i, w, h, y1, y2 in _ARG_VALUES]

_DEBUG = False


def reseed_libc():
    from ctypes import CDLL
    libc = CDLL("libc.so.6")
    libc.srand(12345)


def resize(image, width, height):
    if width is None:
        width = image.width
    if height is None:
        height = image.height
    size = (width, height)
    yield image if image.size == size else image.resize(size)


@pytest.fixture
def mandelbrot(width, height):
    img = Image.effect_mandelbrot((width, height), (-1.5, -1, 0.5, 1), 100)
    yield img


@pytest.fixture
def noise(width, height):
    reseed_libc()
    img = Image.effect_noise((width, height), 200)
    yield img


@pytest.fixture(scope='module')
def amsterdam_image():
    # Module scope so only loaded once, if needed
    filename = 'joel-de-vriend-J_0p_h4ze68-unsplash.jpg'
    yield Image.open(Path(__file__).parent / 'images' / filename)


@pytest.fixture(scope='module')
def bird_image():
    # Module scope so only loaded once, if needed
    filename = 'chris-andrawes-5qw4M9cQCtg-unsplash.jpg'
    yield Image.open(Path(__file__).parent / 'images' / filename)


@pytest.fixture
def bird(bird_image, width, height):
    yield resize(bird_image, width, height)


@pytest.fixture
def amsterdam(amsterdam_image, width, height):
    yield resize(amsterdam_image, width, height)


@pytest.fixture
def debug_dir_path(request, tmp_path_factory):
    """If debugging, yield a new directory for the current test, formed from the test name."""
    if _DEBUG:
        # "test[noise, w=5, h=5, y1=3, y2=1]" -> "test-noise_w5_h5_y13_y21-"
        dir_name = request.node.name
        dir_name = re.sub(r'[\[\]]', '-', dir_name)
        dir_name = re.sub(r'[=,]', '', dir_name)
        dir_name = re.sub(r'\s+', '_', dir_name)
        tmp_path = tmp_path_factory.mktemp(dir_name)
    else:
        tmp_path = None
    yield tmp_path


def _debug_save(debug_dir_path, /, **images):
    """If debugging, save all the given images under debug_dir_path."""
    if _DEBUG:
        for name, img in images.items():
            img.save(debug_dir_path / f'{name}.png')


@pytest.mark.parametrize(argnames=_ARG_NAMES, argvalues=_ARG_VALUES, ids=_IDS)
def test(image_name, width, height, crop_y_1, crop_y_2, debug_dir_path, request):
    """Create two overlapping crops of an image (width x height) and test that stitching them restores the original."""

    # Retrieve an image of size (width x height) from the named pytest fixture
    expected = request.getfixturevalue(image_name)
    if isinstance(expected, Iterator):
        expected = next(expected)

    # Create two crops
    img1 = expected.crop((0, 0, width, crop_y_1))
    img2 = expected.crop((0, crop_y_2, width, height))
    _debug_save(debug_dir_path, expected=expected, img1=img1, img2=img2)

    # Stitch the crops
    actual = stitch(img1, img2)

    _debug_save(debug_dir_path, actual=actual)

    # Verify
    assert actual.size == expected.size
    assert actual.mode == expected.mode
    assert actual.tobytes() == expected.tobytes()
