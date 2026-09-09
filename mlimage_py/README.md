mlimage Python Bindings
=======================

This python extension provides a Python interface for reading MeVisLab's
`.mlimage` file format designed for volumetric medical imaging.  It supports
async I/O and therefore needs to be called from an async context, here using
`ipython --gui=asyncio`:

```python
In [1]: import mlimage, asyncio

In [2]: reader = mlimage.MLImageFormatReader("../assets/test_32x32x8_LZ4.mlimage")

In [3]: reader.image_extent
Out[3]: [32, 32, 8, 1, 1, 1]

In [4]: reader.world_matrix
Out[4]:
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])

In [5]: await reader.get_tile([5, 5, 1, 0, 0, 0], [5 + 2, 5 + 2, 1 + 2, 1, 1, 1])
Out[5]:
array([[[[[[ 9513,  9521],
           [ 9769,  9777]],

          [[17706, 17714],
           [17962, 17970]]]]]], dtype=uint16)
```

The extension contains bindings for the `mlimage-rs` rust crate and just
consists of instantiations of the underlying Rust loader for a few primitive
voxel types.

Caveat
------

The async `get_tile()` method is actually a synchronous wrapper that wraps a
Rust future inside a Python future, and this wrapping needs an event loop. This
means that *one cannot directly call `get_tile` through `asyncio.run`*, see
https://github.com/awestlake87/pyo3-asyncio#a-note-about-asynciorun for details.

The following example does not use `--gui=asyncio`, so that there is no event
loop set up yet:

```python
In [1]: import mlimage, asyncio

In [2]: reader = mlimage.MLImageFormatReader("../assets/test_32x32x8_LZ4.mlimage")

In [3]: asyncio.run(reader.get_tile([0, 0, 0, 0, 0, 0], reader.image_extent))
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[3], line 1
----> 1 asyncio.run(reader.get_tile([0, 0, 0, 0, 0, 0], reader.image_extent))

RuntimeError: no running event loop

In [4]: async def get_whole_image(reader):
   ...:     return await reader.get_tile([0]*6, reader.image_extent)
   ...:

In [5]: data = asyncio.run(get_whole_image(reader))

In [6]: data.shape
Out[6]: (1, 1, 1, 8, 32, 32)
```

License
-------

This project, including the Rust crate and Python bindings, is licensed under
the [Apache 2.0 license](../LICENSE.Apache-2.0).
