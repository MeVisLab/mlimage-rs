Development
===========

This project is developed using the latest stable Rust version by default.

Vector Type
-----------

I am still looking for a suitable vector type that I would prefer over rolling
my own. However, all external crates that I considered seemed to be overkill for
the requirements I found to have:

* statically sized, "cheap" vectors
* at least 3D and 6D, probably we also need 2D or 4D/5D eventually
* maybe even quaternions
* generic over its type, of course
* a box type (pair of start/end) would be great, too
* some information on the dimensions would be cool
  * in particular, we have to deal with C-order and Fortran-order vectors

For now, I have been using plain arrays (`[Ix; 6]` for instance), and used `_c`
suffixes to mark C-order vectors (utczyx), while the default is Fortran order
(xyzctu).

Operations turned out to be relatively concisely implementable using the
`izip!()` macro, and a collect6d() function helps to get from iterators back to
the array:

```rust
    let page_index_start: [Ix; 6] =
        collect6d(izip!(&box_start, &page_extent).map(|(pos, ext)| pos / ext));
```
