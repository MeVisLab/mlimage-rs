use std::{
    fmt::Display,
    str::{from_utf8, FromStr},
};

use winnow::{
    ascii::{dec_uint, digit0, space0},
    binary as wb,
    combinator::{delimited, repeat, terminated},
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

#[derive(Debug)]
pub struct PageIdxEntry {
    pub start_offset: u64,
    pub end_offset: u64,
    pub is_compressed: bool,
    pub checksum: u32, // 24 bit
    pub flag_byte: u8,
    // TODO: remove from PageIdxEntry? Size varies in any case.
    pub default_voxel_value: u16,
}

type PageIdxTable = ndarray::Array6<PageIdxEntry>;

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

#[derive(Debug)]
pub struct MLImage {
    pub version: VersionHeader,
    pub endianess: winnow::binary::Endianness,
    pub tag_list: TagList,
    pub page_idx_table: PageIdxTable,
    pub uses_partial_pages: bool,
    pub world_matrix: ndarray::Array2<f64>,
}

// like dec_uint(), but accepts leading zeros (and is less optimized)
fn parse_uint<T>(input: &mut &[u8]) -> PResult<T>
where
    T: FromStr,
{
    digit0
        .verify_map(|s: &[u8]| {
            std::str::from_utf8(s)
                .expect("we only got ASCII input")
                .parse::<T>()
                .ok()
        })
        .parse_next(input)
}

pub fn version_header(input: &mut &[u8]) -> PResult<VersionHeader> {
    let (_header, major, minor, patch) = (
        "MLImageFormatVersion.",
        terminated(parse_uint::<u16>, '.'),
        terminated(parse_uint::<u16>, '.'),
        terminated(parse_uint::<u16>, '\0'),
    )
        .parse_next(input)?;
    Ok(VersionHeader {
        major,
        minor,
        patch,
    })
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
    endianess: winnow::binary::Endianness,
) -> impl Parser<Input, PageIdxEntry, Error>
where
    Input: StreamIsPartial + Stream<Token = u8>,
    <Input as Stream>::Slice: AsBytes,
    Error: ParserError<Input>,
{
    (
        wb::u64(endianess),
        wb::u64(endianess),
        wb::u8,
        wb::u8,
        wb::u8,
        wb::u8,
        wb::u8,
        repeat::<_, _, Vec<_>, _, _>(11, wb::u8),
        wb::u16(endianess), // FIXME: depends on voxel type!
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
                default_voxel_value,
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
                    default_voxel_value,
                }
            },
        )
}

pub fn parse_file(input: &mut &[u8]) -> PResult<MLImage> {
    let version = version_header.parse_next(input)?;

    // tag_list_size includes this first tag pair, so we need to take a backup of the input:
    let peek_tag_list_input = &mut &input[..];
    let tag_list_size = tag_list_size_in_bytes.parse_next(peek_tag_list_input)?;

    // TODO: is the split_at() + parse() ideomatic?
    // it requires a relatively ugly error wrapping
    let (tag_list_input, input) = input.split_at(tag_list_size);
    let tag_list_input = &mut &(*tag_list_input); // we need to re-establish a mutable cursor after the split_at()
    let input = &mut &(*input); // we need to re-establish a mutable cursor after the split_at()

    let tag_list = TagList(tag_list.parse_next(tag_list_input)?);

    let endianess = if tag_list
        .parse_tag_value::<u8>("ML_ENDIANESS")
        .map_err(|e| ErrMode::from_external_error(&tag_list_input, ErrorKind::Tag, e))?
        > 0
    {
        winnow::binary::Endianness::Big
    } else {
        winnow::binary::Endianness::Little
    };

    let dtype_size: usize = tag_list.parse_tag_value("ML_IMAGE_DTYPE_SIZE").unwrap();
    assert_eq!(dtype_size, 2);

    let image_extent: Vec<usize> = "XYZCTU"
        .chars()
        .filter_map(|dim| {
            tag_list
                .parse_tag_value(&format!("ML_IMAGE_EXT_{}", dim))
                .ok()
        })
        .collect();
    let page_extent: Vec<usize> = "XYZCTU"
        .chars()
        .filter_map(|dim| {
            tag_list
                .parse_tag_value(&format!("ML_PAGE_EXT_{}", dim))
                .ok()
        })
        .collect();
    let page_count_per_dim: Vec<usize> = image_extent
        .iter()
        .zip(page_extent.iter())
        .map(|(ie, pe)| num::Integer::div_ceil(ie, pe))
        .collect();
    let page_count_per_dim: [usize; 6] = page_count_per_dim
        .try_into()
        .map_err(|_| ErrMode::Cut(ContextError::new()))?;
    let total_page_count = page_count_per_dim.iter().product::<usize>();

    let pages: Vec<_> =
        repeat(total_page_count, parse_page_idx_entry(endianess)).parse_next(input)?;

    let page_idx_table = ndarray::Array::from_vec(pages)
        .into_shape(page_count_per_dim)
        .expect("reshaping should not fail");

    let uses_partial_pages = tag_list
        .parse_tag_value::<i8>("ML_USES_PARTIAL_PAGES")
        .map_or(false, |pp| pp > 0);

    let mut world_matrix = ndarray::Array2::zeros((4, 4));
    for row in 0..4 {
        for col in 0..4 {
            world_matrix[(row, col)] = tag_list
                .parse_tag_value(&format!("ML_WORLD_MATRIX_{}{}", row, col))
                .map_err(|e| {
                    ErrMode::Cut(ContextError::from_external_error(
                        &tag_list_input,
                        ErrorKind::Tag,
                        e,
                    ))
                })?;
        }
    }

    Ok(MLImage {
        version,
        endianess,
        tag_list,
        page_idx_table,
        uses_partial_pages,
        world_matrix,
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
        let result = tag_string.parse(b"ML_ENDIANESS\0");
        assert!(result.is_ok());
        if let Some(result_string) = result.ok() {
            assert_eq!(&result_string, "ML_ENDIANESS");
        }
    }

    #[test]
    fn test_tag_pair() {
        let result = tag_pair.parse(b"ML_ENDIANESS\00\0");
        assert!(result.is_ok());
        if let Some((tag_name, tag_value)) = result.ok() {
            assert_eq!(&tag_name, "ML_ENDIANESS");
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
        if let Some(image) = result.ok() {
            assert_eq!(image.version.major, 0);
            assert_eq!(image.version.minor, 1);
        }
    }

    #[test]
    fn test_image_data_uncompressed() {
        let asset = include_bytes!("../assets/test_32x32x8_None.mlimage");
        let result = parse_file.parse_next(&mut &asset[..]);
        assert!(result.is_ok());
        if let Some(image) = result.ok() {
            let idx_entry = &image.page_idx_table[[0, 0, 0, 0, 0, 0]];
            let _raw_data = &asset[idx_entry.start_offset.try_into().unwrap()
                ..idx_entry.end_offset.try_into().unwrap()];
            //let uint16_data:
        }
    }

    #[test]
    fn test_image_data_lz4() {
        let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");
        let result = parse_file.parse_next(&mut &asset[..]);
        assert!(result.is_ok());
        if let Some(image) = result.ok() {
            let idx_entry = &image.page_idx_table[[0, 0, 0, 0, 0, 0]];

            // 2023-09-09: I checked the start/end offsets, and they're really file offsets:
            let raw_data = &mut &asset[idx_entry.start_offset.try_into().unwrap()
                ..idx_entry.end_offset.try_into().unwrap()];

            // TODO: unwrap() -> ?
            let (uncompressed_size, _flags) = ((
                wb::i64::<&[u8], ()>(image.endianess),
                wb::i64(image.endianess),
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
