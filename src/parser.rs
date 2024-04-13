use std::{
    error::Error,
    fmt::Display,
    fs::File,
    io::{BufRead, BufReader, Read, Seek},
    path::Path,
    str::{from_utf8, FromStr},
};

use bytemuck::{cast_slice_mut, from_bytes, Pod};
use ndarray::Ix;
use winnow::{
    ascii::{dec_uint, digit1, space0},
    binary as wb,
    combinator::{delimited, preceded, repeat, terminated},
    error::{ContextError, ErrMode, ErrorKind, FromExternalError, ParserError},
    stream::{AsBytes, Stream, StreamIsPartial},
    token::take_until,
    PResult, Parser,
};

#[derive(Debug)]
pub struct VersionHeader {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
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

#[derive(Debug)]
pub struct TagList(Vec<(String, String)>);

#[derive(Debug, Clone)]
pub struct TagError {
    pub tag_name: String,
    /// None means that the tag was missing, otherwise it could not be parsed
    pub tag_value: Option<String>,
}

impl std::error::Error for TagError {}

impl Display for TagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(value) = &self.tag_value {
            write!(
                f,
                "could not parse tag '{}' with value '{}'",
                self.tag_name, value
            )
        } else {
            write!(f, "tag '{}' not found", self.tag_name)
        }
    }
}

#[derive(Debug, Default)]
pub struct IncompleteFile {}

impl std::error::Error for IncompleteFile {}

impl Display for IncompleteFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unexpected EOF (MLImageFormat file truncated)")
    }
}

#[derive(Debug, Default)]
pub struct InvalidFile {
    msg: String,
}

impl From<ContextError> for InvalidFile {
    fn from(error: ContextError) -> Self {
        Self {
            msg: format!("{}", error),
        }
    }
}

impl std::error::Error for InvalidFile {}

impl Display for InvalidFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid MLImageFormat ({})", self.msg)
    }
}

impl TagList {
    pub fn tag_value(&self, tag_name: &str) -> Option<String> {
        self.0.iter().find_map(|(key, value)| {
            if key == tag_name {
                Some(value.clone())
            } else {
                None
            }
        })
    }

    pub fn parse_tag_value<F: FromStr>(&self, tag_name: &str) -> Result<F, TagError> {
        if let Some(value) = self.tag_value(tag_name) {
            value.parse().or(Err(TagError {
                tag_name: tag_name.to_owned(),
                tag_value: Some(value),
            }))
        } else {
            Err(TagError {
                tag_name: tag_name.to_owned(),
                tag_value: None,
            })
        }
    }
}

pub struct MLImageFormatReader {
    version: VersionHeader,
    info: MLImageInfo,
    reader: BufReader<File>,
    page_idx_table_start: u64,
}

impl MLImageFormatReader {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let f = File::open(path)?;
        let mut reader = BufReader::with_capacity(64 * 1024, f);
        reader.fill_buf()?;

        let (version, tag_list_size) = (version_header, tag_list_size_in_bytes)
            .parse_next(&mut reader.buffer())
            .map_err(|_| IncompleteFile::default())?;
        let page_idx_table_start = VERSION_HEADER_SIZE + tag_list_size;

        let info = if reader.capacity() >= page_idx_table_start {
            parse_file.parse_next(&mut reader.buffer())
        } else {
            reader.rewind()?;
            let mut header_buf = Vec::with_capacity(page_idx_table_start);
            reader.read_exact(&mut header_buf)?;
            parse_file.parse_next(&mut &header_buf[..])
        }
        .map_err(|err_mode| {
            InvalidFile::from(
                err_mode
                    .into_inner()
                    .expect("parser should not return incomplete"),
            )
        })?;

        Ok(Self {
            version,
            info,
            reader,
            page_idx_table_start: page_idx_table_start as u64,
        })
    }

    pub fn get_page_idx_entry(&mut self, index: [Ix; 6]) -> Result<&PageIdxEntry, Box<dyn Error>> {
        // if let Some(page_idx_entry) = self.info.page_idx_table[index].as_ref() {
        //     return Ok(page_idx_entry);
        // }

        if self.info.page_idx_table[index].is_none() {
            let mut flat_page_index = 0;
            for dim in 0..6 {
                flat_page_index *= self.info.page_idx_table.shape()[dim];
                flat_page_index += index[dim];
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

            self.info.page_idx_table[index] = Some(page_idx_entry);
        }

        Ok(&self.info.page_idx_table[index].as_ref().unwrap())
    }

    pub fn read_page<VoxelType>(&mut self, index: [Ix; 6]) -> Result<ndarray::Array6<VoxelType>, Box<dyn Error>>
    where
        VoxelType: Default + Pod,
    {
        let page_idx_entry = self.get_page_idx_entry([0, 0, 0, 0, 0, 0])?.clone();
        let default_voxel_value: VoxelType = *from_bytes(&page_idx_entry.raw_voxel_value[..]);
        let mut result =
            ndarray::Array6::<VoxelType>::from_elem(self.info.page_extent, default_voxel_value);
        self.reader.seek(std::io::SeekFrom::Start(page_idx_entry.start_offset.try_into().unwrap()))?;
        let read_size = page_idx_entry.end_offset - page_idx_entry.start_offset;
        let target_voxeltype_buf = result.as_slice_mut().expect("freshly constructed array should be contiguous");
        let target_u8_buf: &mut [u8] = cast_slice_mut(target_voxeltype_buf);
        let compressor_name = self.info.tag_list.tag_value("ML_COMPRESSOR_NAME").unwrap_or(String::default());
        match &compressor_name[..] {
            "" => {
                assert!(target_u8_buf.len() == read_size.try_into().unwrap());
                self.reader.read_exact(target_u8_buf)?;
            },
            _ => {
                todo!("compressor '{}' not implemented yet", &compressor_name);
            }
        }
        
        Ok(result)
    }
}

#[derive(Debug)]
pub struct MLImageInfo {
    pub endianness: winnow::binary::Endianness,
    pub dtype_size: usize,
    pub image_extent: [Ix; 6],
    pub page_extent: [Ix; 6],
    pub tag_list: TagList,
    pub page_idx_table: PageIdxTable,
    pub uses_partial_pages: bool,
    pub world_matrix: ndarray::Array2<f64>,
}

impl MLImageInfo {
    pub fn from_tag_list(tag_list: TagList) -> Result<Self, TagError> {
        let endianness = if tag_list.parse_tag_value::<u8>("ML_ENDIANESS")? > 0 {
            winnow::binary::Endianness::Big
        } else {
            winnow::binary::Endianness::Little
        };

        let dtype_size: usize = tag_list.parse_tag_value("ML_IMAGE_DTYPE_SIZE").unwrap();

        let image_extent: Vec<Ix> = "XYZCTU"
            .chars()
            .filter_map(|dim| {
                tag_list
                    .parse_tag_value(&format!("ML_IMAGE_EXT_{}", dim))
                    .ok()
            })
            .collect();
        let image_extent: [Ix; 6] = image_extent
            .try_into()
            .expect("by construction, we must have 6D extents");

        let page_extent: Vec<usize> = "XYZCTU"
            .chars()
            .filter_map(|dim| {
                tag_list
                    .parse_tag_value(&format!("ML_PAGE_EXT_{}", dim))
                    .ok()
            })
            .collect();
        let page_extent: [Ix; 6] = page_extent
            .try_into()
            .expect("by construction, we must have 6D extents");

        let page_count_per_dim: Vec<usize> = image_extent
            .iter()
            .zip(page_extent.iter())
            .map(|(ie, pe)| num::Integer::div_ceil(ie, pe))
            .collect();
        let page_count_per_dim: [usize; 6] = page_count_per_dim
            .try_into()
            .expect("by construction, we must have 6D extents");

        let page_idx_table = ndarray::Array::from_elem(page_count_per_dim, None);

        let uses_partial_pages = tag_list
            .parse_tag_value::<i8>("ML_USES_PARTIAL_PAGES")
            .map_or(false, |pp| pp > 0);

        let mut world_matrix = ndarray::Array2::zeros((4, 4));
        for row in 0..4 {
            for col in 0..4 {
                world_matrix[(row, col)] =
                    tag_list.parse_tag_value(&format!("ML_WORLD_MATRIX_{}{}", row, col))?;
            }
        }

        Ok(Self {
            endianness,
            dtype_size,
            image_extent,
            page_extent,
            tag_list,
            page_idx_table,
            uses_partial_pages,
            world_matrix,
        })
    }
}

const VERSION_HEADER_SIZE: usize = "MLImageFormatVersion.".len() + 3 * 4;
const PAGE_IDX_ENTRY_SIZE: usize = 8 + 8 + 5 + 11;

pub fn version_header(input: &mut &[u8]) -> PResult<VersionHeader> {
    preceded(
        "MLImageFormatVersion.",
        (
            terminated(digit1.parse_to::<u16>(), b'.'),
            terminated(digit1.parse_to::<u16>(), b'.'),
            terminated(digit1.parse_to::<u16>(), b'\0'),
        ),
    )
    .map(|(major, minor, patch)| VersionHeader {
        major,
        minor,
        patch,
    })
    .parse_next(input)
}

pub fn tag_string(input: &mut &[u8]) -> PResult<String> {
    terminated(take_until(0.., 0u8), 0u8)
        .try_map(|buf: &[u8]| from_utf8(buf).map(|s| s.to_owned()))
        .parse_next(input)
}

pub fn tag_pair(input: &mut &[u8]) -> PResult<(String, String)> {
    (tag_string, tag_string).parse_next(input)
}

pub fn tag_list_size_in_bytes(input: &mut &[u8]) -> PResult<usize> {
    delimited(
        "ML_TAG_LIST_SIZE_IN_BYTES\0",
        dec_uint::<_, usize, _>,
        (space0, 0u8),
    )
    .parse_next(input)
}

pub fn tag_list(input: &mut &[u8]) -> PResult<Vec<(String, String)>> {
    repeat(0.., tag_pair).parse_next(input)
}

// see http://mevislabdownloads.mevis.de/docs/current/MeVisLab/Resources/Documentation/Publish/SDK/ToolBoxReference/mlImageFormatIdxTable_8h_source.html
fn parse_page_idx_entry<Input, Error>(
    endianness: winnow::binary::Endianness,
    dtype_size: usize,
) -> impl Parser<Input, PageIdxEntry, Error>
where
    Input: StreamIsPartial + Stream<Token = u8>,
    <Input as Stream>::Slice: AsBytes,
    Error: ParserError<Input>,
{
    (
        wb::u64(endianness),
        wb::u64(endianness),
        wb::u8,
        wb::u8,
        wb::u8,
        wb::u8,
        wb::u8,
        repeat::<_, _, Vec<_>, _, _>(11, wb::u8).void(),
        repeat(dtype_size, wb::u8),
    )
        .map(
            |(
                start_offset,
                end_offset,
                is_compressed,
                checksum_low,
                checksum_med,
                checksum_high,
                flag_byte,
                _internal,
                raw_voxel_value,
            )| {
                let is_compressed = is_compressed > 0;
                let checksum = (checksum_low as u32)
                    | (checksum_med as u32) << 8
                    | (checksum_high as u32) << 16;
                PageIdxEntry {
                    start_offset,
                    end_offset,
                    is_compressed,
                    checksum,
                    flag_byte,
                    raw_voxel_value,
                }
            },
        )
}

pub fn parse_file(input: &mut &[u8]) -> PResult<MLImageInfo> {
    let _version = version_header.parse_next(input)?;

    // tag_list_size includes this first tag pair, so we need to checkpoint the input:
    let tag_list_begin = input.checkpoint();
    let tag_list_size = tag_list_size_in_bytes.parse_next(input)?;
    input.reset(&tag_list_begin);

    let tag_list_buffer: Vec<_> = repeat(tag_list_size, wb::u8).parse_next(input)?;
    let tag_list = TagList(tag_list.parse_next(&mut &tag_list_buffer[..])?);

    MLImageInfo::from_tag_list(tag_list).map_err(|e| {
        ErrMode::Cut(ContextError::from_external_error(
            &tag_list_begin,
            ErrorKind::Tag,
            e,
        ))
    })
}

#[cfg(test)]
mod tests {
    use lz4_flex::decompress;

    use super::*;

    #[test]
    fn test_version_string() {
        let result = version_header.parse(b"MLImageFormatVersion.000.001.000\0");
        assert!(result.is_ok());
        if let Some(result_version) = result.ok() {
            assert_eq!(result_version.major, 0);
            assert_eq!(result_version.minor, 1);
            assert_eq!(result_version.patch, 0);
        }
    }

    #[test]
    fn test_tag_string() {
        let result = tag_string.parse(b"ML_ENDIANESS\0"); // sic
        assert!(result.is_ok());
        if let Some(result_string) = result.ok() {
            assert_eq!(&result_string, "ML_ENDIANESS"); // sic
        }
    }

    #[test]
    fn test_tag_pair() {
        let result = tag_pair.parse(b"ML_ENDIANESS\00\0"); // sic
        assert!(result.is_ok());
        if let Some((tag_name, tag_value)) = result.ok() {
            assert_eq!(&tag_name, "ML_ENDIANESS"); // sic
            assert_eq!(&tag_value, "0");
        }
    }

    #[test]
    fn test_tag_list_size_in_bytes() {
        let result = tag_list_size_in_bytes.parse(b"ML_TAG_LIST_SIZE_IN_BYTES\01115           \0");
        dbg!(&result);
        assert!(result.is_ok());
        if let Some(size) = result.ok() {
            assert_eq!(size, 1115);
        }
    }

    #[test]
    fn test_reading_missing_tag() {
        let tag_list = TagList(vec![("SOME_TAG".to_string(), "existing_value".to_string())]);
        assert!(tag_list.tag_value("MISSING_TAG").is_none());
        assert!(tag_list.parse_tag_value::<usize>("MISSING_TAG").is_err());
    }

    #[test]
    fn test_reading_int_tag() {
        let tag_list = TagList(vec![("SOME_TAG".to_string(), "123".to_string())]);
        assert!(tag_list.tag_value("SOME_TAG").is_some());
        assert_eq!(tag_list.tag_value("SOME_TAG").unwrap(), "123");
        assert!(tag_list.parse_tag_value::<usize>("SOME_TAG").is_ok());
        assert_eq!(tag_list.parse_tag_value::<usize>("SOME_TAG").unwrap(), 123);
    }

    #[test]
    fn test_tag_list() {
        let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");
        let tag_list_buf = &asset[0x21..0x21 + 1211];
        let result = tag_list.parse(tag_list_buf);
        if let Err(e) = &result {
            hexdump::hexdump(e.input());
            println!("Error at position {:#04x}", e.offset());
            assert!(false);
        }
    }

    #[test]
    fn test_file() {
        let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");
        let result = parse_file.parse_next(&mut &asset[..]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_image_data_uncompressed() {
        let result = MLImageFormatReader::open("assets/test_32x32x8_None.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let result_page_buf = reader.read_page::<u16>([0, 0, 0, 0, 0, 0]);
            assert!(result_page_buf.is_ok());
        }
    }

    #[test]
    fn test_image_data_lz4() {
        let result = MLImageFormatReader::open("assets/test_32x32x8_LZ4.mlimage");
        assert!(result.is_ok());
        if let Some(mut reader) = result.ok() {
            let idx_entry = reader.get_page_idx_entry([0, 0, 0, 0, 0, 0]).unwrap();

            let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");

            // 2023-09-09: I checked the start/end offsets, and they're really file offsets:
            let raw_data = &mut &asset[idx_entry.start_offset.try_into().unwrap()
                ..idx_entry.end_offset.try_into().unwrap()];

            // TODO: unwrap() -> ?
            let (uncompressed_size, _flags) = ((
                wb::i64::<&[u8], ()>(reader.info.endianness),
                wb::i64(reader.info.endianness),
            ))
                .parse_next(raw_data)
                .unwrap();

            let decompressed = decompress(raw_data, usize::try_from(uncompressed_size).unwrap())
                .expect("decompression failed");
            //let mut decomp = lz4_flex::frame::FrameDecoder::new(raw_data);
            //let mut decompressed: Vec<u8> = Vec::new();
            //std::io::copy(&mut decomp, &mut decompressed).expect("decompression failed");
            dbg!(decompressed);
        }
    }
}
