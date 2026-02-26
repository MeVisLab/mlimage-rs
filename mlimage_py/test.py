import numpy as np
import mlimage
import timeit
import asyncio
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_read_mlimage_header():
    reader = mlimage.MLImageFormatReader("../assets/test_32x32x8_LZ4.mlimage")
    assert reader.image_extent == [32, 32, 8, 1, 1, 1]
    assert reader.page_extent == [16, 16, 2, 1, 1, 1]
    assert np.allclose(reader.world_matrix, np.eye(4))


@pytest.mark.asyncio
async def test_read_mlimage_roi():
    reader = mlimage.MLImageFormatReader("../assets/test_32x32x8_LZ4.mlimage")
    roi = await reader.get_tile([5, 5, 1, 0, 0, 0], [5 + 20, 5 + 20, 1 + 2, 1, 1, 1])
    assert roi.shape == (1, 1, 1, 2, 20, 20)
    assert roi.dtype == np.uint16
    # value range is 0..65535, but the ROI only contains values in a smaller range
    # (the following range was determined using MeVisLab)
    assert roi.min() == 9513
    assert roi.max() == 22722


@pytest.mark.asyncio
async def test_read_mlimage_fully():
    reader = mlimage.MLImageFormatReader("../assets/test_32x32x8_LZ4.mlimage")
    image = await reader.get_tile([0] * 6, reader.image_extent)
    assert image.shape == (1, 1, 1, 8, 32, 32)
    assert image.dtype == np.uint16
    assert image.max() == 65535
    assert image.min() == 0
