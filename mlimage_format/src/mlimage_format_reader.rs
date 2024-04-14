use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read, Seek},
    path::Path,
};

use bytemuck::{cast_slice_mut, from_bytes, Pod};
use ndarray::Ix;
use winnow::{
    binary as wb,
    error::{ContextError, ErrMode},
    Parser,
};

use crate::{
    errors::{IncompleteFile, InvalidFile},
    mlimage_info::MLImageInfo,
    parser::{parse_info, parse_page_idx_entry, tag_list_size_in_bytes, version_header},
};

#[derive(Debug)]
pub struct VersionHeader {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

pub struct MLImageFormatReader {
    version: VersionHeader,
    info: MLImageInfo,
    reader: BufReader<File>,
    page_idx_table_start: u64,
    pub page_idx_table: PageIdxTable,
}

#[derive(Debug, Clone)]
pub struct PageIdxEntry {
    pub start_offset: u64,
    pub end_offset: u64,
    pub is_compressed: bool,
    pub checksum: u32, // 24 bit
    pub flag_byte: u8,
    pub raw_voxel_value: Vec<u8>,
}

type PageIdxTable = ndarray::Array6<Option<PageIdxEntry>>;

const VERSION_HEADER_SIZE: usize = "MLImageFormatVersion.".len() + 3 * 4;
const PAGE_IDX_ENTRY_SIZE: usize = 8 + 8 + 5 + 11;

impl MLImageFormatReader {
    pub fn info(&self) -> &MLImageInfo {
        &self.info
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let f = File::open(path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, f);
        reader.fill_buf()?;

        let (version, tag_list_size) = (version_header, tag_list_size_in_bytes)
            .parse_next(&mut reader.buffer())
            .map_err(|_| IncompleteFile::default())?;
        let page_idx_table_start = VERSION_HEADER_SIZE + tag_list_size;

        let info = if reader.capacity() >= page_idx_table_start {
            parse_info.parse_next(&mut reader.buffer())
        } else {
            reader.rewind()?;
            let mut header_buf = Vec::with_capacity(page_idx_table_start);
            reader.read_exact(&mut header_buf)?;
            parse_info.parse_next(&mut &header_buf[..])
        }
        .map_err(|err_mode| {
            InvalidFile::from(
                err_mode
                    .into_inner()
                    .expect("parser should not return incomplete"),
            )
        })?;

        let page_idx_table = ndarray::Array::from_elem(info.page_count_per_dim(), None);

        Ok(Self {
            version,
            info,
            reader,
            page_idx_table_start: page_idx_table_start as u64,
            page_idx_table,
        })
    }

    pub fn get_page_idx_entry(&mut self, index: [Ix; 6]) -> Result<&PageIdxEntry, Box<dyn Error>> {
        if self.page_idx_table[index].is_none() {
            let mut flat_page_index = 0;
            for dim in 0..6 {
                flat_page_index *= self.page_idx_table.shape()[5 - dim];
                flat_page_index += index[5 - dim];
            }
            let page_idx_entry_size = PAGE_IDX_ENTRY_SIZE + self.info.dtype_size;

            // TODO: discards buffer; seek_relative should be more efficient:
            self.reader.seek(std::io::SeekFrom::Start(
                self.page_idx_table_start + flat_page_index as u64 * page_idx_entry_size as u64,
            ))?;
            let mut buf = bytes::BytesMut::zeroed(page_idx_entry_size);
            self.reader.read_exact(&mut buf[..])?;

            let page_idx_entry = parse_page_idx_entry(self.info.endianness, self.info.dtype_size)
                .parse_next(&mut &buf[..])
                .map_err(|e: ErrMode<ContextError>| {
                    InvalidFile::from(e.into_inner().expect("not expecting incomplete input here"))
                })?;

            self.page_idx_table[index] = Some(page_idx_entry);
        }

        Ok(&self.page_idx_table[index].as_ref().unwrap())
    }

    pub fn read_page<VoxelType>(
        &mut self,
        index: [Ix; 6],
    ) -> Result<ndarray::Array6<VoxelType>, Box<dyn Error>>
    where
        VoxelType: Default + Pod,
    {
        let page_idx_entry = self.get_page_idx_entry(index)?.clone();
        let default_voxel_value: VoxelType = *from_bytes(&page_idx_entry.raw_voxel_value[..]);

        let mut page_extent_c = self.info.page_extent_c();
        if self.info.uses_partial_pages {
            let image_extent_c = self.info.image_extent_c();
            for dim in 0..6 {
                let start_pos = index[5 - dim] * page_extent_c[dim];
                if image_extent_c[dim] < start_pos + page_extent_c[dim] {
                    page_extent_c[dim] = image_extent_c[dim] - start_pos;
                }
            }
        }

        let mut result =
            ndarray::Array6::<VoxelType>::from_elem(page_extent_c, default_voxel_value);

        // constant pages are stored in an optimized way
        if page_idx_entry.start_offset < page_idx_entry.end_offset {
            self.reader.seek(std::io::SeekFrom::Start(
                page_idx_entry.start_offset.try_into().unwrap(),
            ))?;
            let read_size = page_idx_entry.end_offset - page_idx_entry.start_offset;
            let target_voxeltype_buf = result
                .as_slice_mut()
                .expect("freshly constructed array should be contiguous");
            let target_u8_buf: &mut [u8] = cast_slice_mut(target_voxeltype_buf);
            let compressor_name = self
                .info
                .tag_list
                .tag_value("ML_COMPRESSOR_NAME")
                .unwrap_or(String::default());
            match &compressor_name[..] {
                "" => {
                    assert!(target_u8_buf.len() == read_size.try_into().unwrap());
                    self.reader.read_exact(target_u8_buf)?;
                }
                "LZ4" => {
                    // 2023-09-09: I checked the start/end offsets, and they're really file offsets:
                    let mut buf = bytes::BytesMut::zeroed(
                        read_size
                            .try_into()
                            .expect("raw page size should fit into usize"),
                    );
                    self.reader.read_exact(&mut buf[..])?;

                    // TODO: unwrap() -> ?
                    let (uncompressed_size, flags) = ((
                        wb::i64::<&[u8], ()>(wb::Endianness::Little),
                        wb::i64(wb::Endianness::Little),
                    ))
                        .parse_next(&mut &buf[..])
                        .unwrap();
                    let uncompressed_size = usize::try_from(uncompressed_size)
                        .expect("uncompressed page size should fit into usize");

                    let byte_plane_reordering = (flags & 1) > 0;
                    let diff_code_data = (flags & 2) > 0;

                    assert!(target_u8_buf.len() == uncompressed_size);

                    // TODO: what's the meaning of the return code? its a usize that
                    // is smaller than uncompressed_size, but larger than read_size
                    // (and not exactly the difference)
                    let _decompressed = lz4_flex::decompress_into(&buf[16..], target_u8_buf)
                        .expect("decompression failed");

                    //dbg!(byte_plane_reordering, diff_code_data);
                }
                _ => {
                    todo!("compressor '{}' not implemented yet", &compressor_name);
                }
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_data_uncompressed() {
        let result = MLImageFormatReader::open("../assets/test_32x32x8_None.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let result_page_buf = reader.read_page::<u16>([0, 0, 0, 0, 0, 0]);
            assert!(result_page_buf.is_ok());
        }
    }

    #[test]
    fn test_image_data_lz4() {
        let result = MLImageFormatReader::open("../assets/test_32x32x8_LZ4.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let result_page_buf = reader.read_page::<u16>([0, 0, 0, 0, 0, 0]);
            assert!(result_page_buf.is_ok());
        }
    }

    #[test]
    fn test_reading_partial_pages() {
        let result = MLImageFormatReader::open("../assets/test_32x32x8_partial_pages.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let result_page_buf = reader.read_page::<u16>([1, 1, 2, 0, 0, 0]);
            assert!(result_page_buf.is_ok());
        }
    }

    #[test]
    fn test_reading_constant_pages() {
        let result = MLImageFormatReader::open("../assets/test_32x32x8_constant_pages.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let result_page_buf = reader.read_page::<u16>([0, 1, 0, 0, 0, 0]);
            assert!(result_page_buf.is_ok());
        }
    }
}
