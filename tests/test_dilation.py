import numpy as np

from fake_bevel_baker.dilation import dilate


def _simple_setup():
    H, W = 32, 32
    img = np.zeros((H, W, 3), dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    # A single valid texel at (16, 16) with value (1, 2, 3)
    img[16, 16] = (1.0, 2.0, 3.0)
    valid[16, 16] = True
    return img, valid


def test_valid_texels_unchanged():
    img, valid = _simple_setup()
    out, out_valid = dilate(img, valid, radius_px=4)
    np.testing.assert_array_equal(out[16, 16], img[16, 16])
    assert out_valid[16, 16]


def test_pixels_within_radius_get_nearest():
    img, valid = _simple_setup()
    out, out_valid = dilate(img, valid, radius_px=4)
    # Pixel at (15, 16) is distance 1 from (16, 16); should be filled.
    assert out_valid[15, 16]
    np.testing.assert_array_equal(out[15, 16], (1.0, 2.0, 3.0))
    # Pixel at (12, 16) is distance 4; on the boundary, should be filled.
    assert out_valid[12, 16]
    # Pixel at (11, 16) is distance 5; outside radius=4.
    assert not out_valid[11, 16]
    np.testing.assert_array_equal(out[11, 16], (0.0, 0.0, 0.0))


def test_radius_zero_returns_copy():
    img, valid = _simple_setup()
    out, out_valid = dilate(img, valid, radius_px=0)
    np.testing.assert_array_equal(out, img)
    np.testing.assert_array_equal(out_valid, valid)
    # Must be a copy, not the same object
    assert out is not img


def test_two_islands_merge_correctly():
    H, W = 16, 16
    img = np.zeros((H, W, 3), dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    img[4, 4] = (1.0, 0.0, 0.0)
    valid[4, 4] = True
    img[4, 12] = (0.0, 1.0, 0.0)
    valid[4, 12] = True

    out, out_valid = dilate(img, valid, radius_px=3)
    # Midpoint (4, 8) is 4 from each island; outside radius=3.
    assert not out_valid[4, 8]
    # (4, 5) is 1 from island A, 7 from B -> gets A
    np.testing.assert_array_equal(out[4, 5], (1.0, 0.0, 0.0))
    # (4, 11) is 7 from A, 1 from B -> gets B
    np.testing.assert_array_equal(out[4, 11], (0.0, 1.0, 0.0))


def test_grayscale_image_works():
    valid = np.zeros((8, 8), dtype=bool)
    valid[3, 3] = True
    img = np.zeros((8, 8), dtype=np.float32)
    img[3, 3] = 42.0
    out, out_valid = dilate(img, valid, radius_px=2)
    assert out.shape == img.shape
    assert out[3, 4] == 42.0
    assert out_valid[3, 4]
