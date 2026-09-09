# mlimage-rs

This is a Rust implementation of the MLImage file format that was designed and
implemented for MeVisLab's 6D medical image volumes. For now, this is an
experiment to see how far one can get, and how an interface that is similar to
ML's (MeVisLab library) API can be implemented in Rust.

There are many things that are unfinished, incomplete, and not perfect, but at
least as of 2024-04-14, I could already read 16bit image data from an .mlimage
file in python, although still quite clumsily.

As of 2024-10-05, there is also a proper `get_tile()` implementation in Rust
that makes it much less clumsy.

Build instructions can be found in [the DEVELOPMENT.md documentation](DEVELOPMENT.md).

## Supported .mlimage Features

Reading of MLImages is implemented using generic Rust functions, and a Python
wrapper makes it easy to read those files in Python using a number of primitive
voxel types.  I would expect the following to work in general; for most of these
features, some unit tests exist (either on the Rust or Python level).
In other words, please report back if something does not work as expected yet:

- opening MLImages, querying metadata (without reading the whole file)
- caching the page table for efficient reading
  (but handle extremely large tables efficiently as well by not reading all at once)
- reading of single pages
- get_tile for arbitrary boxes
- async Rust (and Python) for efficient, async IO (considering a hypothetical
  future with a Rust-based, ML-like, multi-threaded image processing runtime)
- file format features
  - different primitive voxel types
    - unsigned and signed integers
    - bit widths of 8, 16, 32, 64
    - float / double
  - sparse files (constant pages)
  - uncompressed files
  - LZ4 compression
  - byte plane reordering (for better compression)

## Limitations

Currently, many things are not yet implemented:

- currently, only _reading_ is possible, no writing yet
- only the LZ4 compressor is implemented (and uncompressed files are supported)
- diff (de)coding not implemented yet
- no dynamic loading of plugins (the original C++ implementation dynamically
  loads compressors, for instance), yet that may not be necessary in practice

I would also like to improve the *how* in the sense that I would like to have
cleaner Rust abstractions for vectors and possibly page-based images in general.

## References

https://mevislabdownloads.mevis.de/docs/current/MeVisLab/Resources/Documentation/Publish/SDK/ToolBoxReference/mlImageFormatTagList_8h_source.html
