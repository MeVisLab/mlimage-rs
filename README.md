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

## Limitations

Currently, many things are not yet implemented:

- currently, only _reading_ is possible, no writing yet
- only the LZ4 compressor is implemented (and uncompressed files are supported)
- no dynamic loading of plugins (the original C++ implementation dynamically
  loads compressors, for instance)

## References

https://mevislabdownloads.mevis.de/docs/current/MeVisLab/Resources/Documentation/Publish/SDK/ToolBoxReference/mlImageFormatTagList_8h_source.html
