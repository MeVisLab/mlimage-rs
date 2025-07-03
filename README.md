# mlimage-rs

This is a Rust implementation of the MLImage file format that is designed and
implemented for MeVisLab's 6D medical image volumes.  For now, this is an
experiment to see how far one can get, and how an interface that is similar to
ML's (MeVisLab library) API can be implemented in Rust.

There are many things that are unfinished, incomplete, and not perfect, but at
least as of 2024-04-14, I could already read 16bit image data from an .mlimage
file in python, although still quite clumsily.

References
----------

https://mevislabdownloads.mevis.de/docs/current/MeVisLab/Resources/Documentation/Publish/SDK/ToolBoxReference/mlImageFormatTagList_8h_source.html
