import pytest
from PIL import Image

from imgstitch.main import ImgStitchError, stitch


def test_stitch_zero():
    with pytest.raises(ValueError):
        stitch()


def test_stitch_one():
    image1 = Image.new('L', (1, 1))
    with pytest.raises(ValueError):
        stitch(image1)


def test_stitch_crop_header_zero():
    image1 = Image.new('L', (1, 1))
    with pytest.raises(ValueError):
        stitch(image1, image1, crop_header_height=-1)


def test_incompatible_width():
    image1 = Image.new('L', (256, 128))
    image2 = Image.new('L', (128, 128))
    with pytest.raises(ImgStitchError):
        stitch(image1, image2)


@pytest.mark.parametrize(
    argnames=['args', 'expected_args', 'expected_exit_code'],
    argvalues=[
        [('message',), ('message',), 2],
        [(), (), 2],
        [(None,), (), 2],
        [(None, 1), (), 1],
        [('', 1), ('',), 1],
    ]
)
def test_exception(args, expected_args, expected_exit_code):
    ex = ImgStitchError(*args)
    assert ex.args == expected_args
    assert ex.exit_code == expected_exit_code
