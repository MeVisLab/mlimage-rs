use std::{
    fmt::Display,
    str::{from_utf8, FromStr},
};

use nom::{
    bytes::complete::{is_not, tag},
    character::complete as cc,
    combinator::{map, opt},
    error::{Error, ErrorKind},
    multi::{count, many0},
    number::complete as nc,
    sequence::{pair, preceded, terminated, tuple},
    Err, IResult,
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
    pub endianess: nom::number::Endianness,
    pub tag_list: TagList,
    pub page_idx_table: PageIdxTable,
    pub uses_partial_pages: bool,
    pub world_matrix: ndarray::Array2<f64>,
}

pub fn version_header(input: &[u8]) -> IResult<&[u8], VersionHeader> {
    let (input, _header) = tag("MLImageFormatVersion.")(input)?;
    let (input, major) = terminated(cc::u16, cc::char('.'))(input)?;
    let (input, minor) = terminated(cc::u16, cc::char('.'))(input)?;
    let (input, patch) = terminated(cc::u16, cc::char('\0'))(input)?;
    Ok((
        input,
        VersionHeader {
            major,
            minor,
            patch,
        },
    ))
}

pub fn tag_string(input: &[u8]) -> IResult<&[u8], String> {
    terminated(
        map(opt(is_not("\0")), |buf| {
            from_utf8(buf.unwrap_or_default()).unwrap().to_owned()
        }),
        tag(b"\0"),
    )(input)
}

pub fn tag_pair(input: &[u8]) -> IResult<&[u8], (String, String)> {
    pair(tag_string, tag_string)(input)
}

pub fn tag_list_size_in_bytes(input: &[u8]) -> IResult<&[u8], usize> {
    preceded(
        tag("ML_TAG_LIST_SIZE_IN_BYTES\0"),
        terminated(
            map(cc::u64, |size| size as _),
            pair(cc::space0, cc::char('\0')),
        ),
    )(input)
}

pub fn tag_list(input: &[u8]) -> IResult<&[u8], Vec<(String, String)>> {
    many0(tag_pair)(input)
}

// see http://mevislabdownloads.mevis.de/docs/current/MeVisLab/Resources/Documentation/Publish/SDK/ToolBoxReference/mlImageFormatIdxTable_8h_source.html
pub fn parse_page_idx_entry(
    endianess: nom::number::Endianness,
    input: &[u8],
) -> IResult<&[u8], PageIdxEntry> {
    let (input, (start_offset, end_offset, is_compressed)) =
        tuple((nc::u64(endianess), nc::u64(endianess), nc::u8))(input)?;
    let is_compressed = is_compressed > 0;

    let (input, (checksum_low, checksum_med, checksum_high, flag_byte)) =
        tuple((nc::u8, nc::u8, nc::u8, nc::u8))(input)?;
    let checksum =
        (checksum_low as u32) | (checksum_med as u32) << 8 | (checksum_high as u32) << 16;

    let (input, internal) = count(nc::u8, 11)(input)?;
    let _internal: [u8; 11] = internal.try_into().unwrap();

    // FIXME: depends on voxel type!
    let (input, default_voxel_value) = nc::u16(endianess)(input)?;

    Ok((
        input,
        PageIdxEntry {
            start_offset,
            end_offset,
            is_compressed,
            checksum,
            flag_byte,
            default_voxel_value,
        },
    ))
}

pub fn parse_file(input: &[u8]) -> IResult<&[u8], MLImage> {
    let (tag_list_input, version) = version_header(input)?;

    let (_input, tag_list_size) = tag_list_size_in_bytes(tag_list_input)?;

    let (tag_list_input, input) = tag_list_input.split_at(tag_list_size);
    let (_nothing, tag_list) = tag_list(tag_list_input)?;
    let tag_list = TagList(tag_list);

    let endianess = if tag_list
        .parse_tag_value::<u8>("ML_ENDIANESS")
        .map_err(|_| Err::Error(Error::new(input, ErrorKind::Tag)))?
        > 0
    {
        nom::number::Endianness::Big
    } else {
        nom::number::Endianness::Little
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
        .map_err(|_| Err::Error(Error::new(input, ErrorKind::Tag)))?;
    let total_page_count = page_count_per_dim.iter().product::<usize>();

    let mut pages = Vec::new();
    let mut input = input;
    for _i in 0..total_page_count {
        let (rest, page_idx_entry) = parse_page_idx_entry(endianess, input)?;
        pages.push(page_idx_entry);
        input = rest;
    }
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
                .map_err(|_| Err::Error(Error::new(input, ErrorKind::Tag)))?;
        }
    }

    Ok((
        input,
        MLImage {
            version,
            endianess,
            tag_list,
            page_idx_table,
            uses_partial_pages,
            world_matrix,
        },
    ))
}

#[cfg(test)]
mod tests {
    use lz4_flex::decompress;

    use super::*;

    #[test]
    fn test_version_string() {
        let result = version_header(b"MLImageFormatVersion.000.001.000\0");
        assert!(result.is_ok());
        if let Some((rest, result_version)) = result.ok() {
            assert_eq!(result_version.major, 0);
            assert_eq!(result_version.minor, 1);
            assert_eq!(result_version.patch, 0);
            assert_eq!(rest.len(), 0);
        }
    }

    #[test]
    fn test_tag_string() {
        let result = tag_string(b"ML_ENDIANESS\0");
        assert!(result.is_ok());
        if let Some((rest, result_string)) = result.ok() {
            assert_eq!(&result_string, "ML_ENDIANESS");
            assert_eq!(rest.len(), 0);
        }
    }

    #[test]
    fn test_tag_pair() {
        let result = tag_pair(b"ML_ENDIANESS\00\0");
        assert!(result.is_ok());
        if let Some((rest, (tag_name, tag_value))) = result.ok() {
            assert_eq!(&tag_name, "ML_ENDIANESS");
            assert_eq!(&tag_value, "0");
            assert_eq!(rest.len(), 0);
        }
    }

    #[test]
    fn test_tag_list_size_in_bytes() {
        let result = tag_list_size_in_bytes(b"ML_TAG_LIST_SIZE_IN_BYTES\01115           \0");
        assert!(result.is_ok());
        if let Some((rest, size)) = result.ok() {
            assert_eq!(size, 1115);
            assert_eq!(rest.len(), 0);
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
    fn test_file() {
        let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");
        let result = parse_file(asset);
        assert!(result.is_ok());
        if let Some((_rest, image)) = result.ok() {
            assert_eq!(image.version.major, 0);
            assert_eq!(image.version.minor, 1);
        }
    }

    #[test]
    fn test_image_data_uncompressed() {
        let asset = include_bytes!("../assets/test_32x32x8_None.mlimage");
        let result = parse_file(asset);
        assert!(result.is_ok());
        if let Some((_rest, image)) = result.ok() {
            let idx_entry = &image.page_idx_table[[0, 0, 0, 0, 0, 0]];
            let _raw_data = &asset[idx_entry.start_offset.try_into().unwrap()
                ..idx_entry.end_offset.try_into().unwrap()];
            //let uint16_data:
        }
    }

    #[test]
    fn test_image_data_lz4() {
        let asset = include_bytes!("../assets/test_32x32x8_LZ4.mlimage");
        let result = parse_file(asset);
        assert!(result.is_ok());
        if let Some((_rest, image)) = result.ok() {
            let idx_entry = &image.page_idx_table[[0, 0, 0, 0, 0, 0]];

            // 2023-09-09: I checked the start/end offsets, and they're really file offsets:
            let raw_data = &asset[idx_entry.start_offset.try_into().unwrap()
                ..idx_entry.end_offset.try_into().unwrap()];

            // TODO: unwrap() -> ?
            let (raw_data, (uncompressed_size, _flags)) =
                tuple((nc::i64::<&[u8], ()>(image.endianess), nc::i64(image.endianess)))(raw_data).unwrap();
                        
            let decompressed = decompress(raw_data, usize::try_from(uncompressed_size).unwrap()).expect("decompression failed");
            //let mut decomp = lz4_flex::frame::FrameDecoder::new(raw_data);
            //let mut decompressed: Vec<u8> = Vec::new();
            //std::io::copy(&mut decomp, &mut decompressed).expect("decompression failed");
            dbg!(decompressed);
        }
    }
}
