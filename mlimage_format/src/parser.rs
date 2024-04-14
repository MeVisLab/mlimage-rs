use std::str::from_utf8;

use winnow::{
    ascii::{dec_uint, digit1, space0},
    binary as wb,
    combinator::{delimited, preceded, repeat, terminated},
    error::{ContextError, ErrMode, ErrorKind, FromExternalError, ParserError},
    stream::{AsBytes, Stream, StreamIsPartial},
    token::take_until,
    PResult, Parser,
};

use crate::{mlimage_format_reader::{PageIdxEntry, VersionHeader}, mlimage_info::MLImageInfo, tag_list::TagList};

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
pub fn parse_page_idx_entry<Input, Error>(
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

pub fn parse_info(input: &mut &[u8]) -> PResult<MLImageInfo> {
    let _version = version_header.parse_next(input)?;

    // tag_list_size includes this first tag pair, so we need to checkpoint the input:
    let tag_list_begin = input.checkpoint();
    let tag_list_size = tag_list_size_in_bytes.parse_next(input)?;
    input.reset(&tag_list_begin);

    let tag_list_buffer: Vec<_> = repeat(tag_list_size, wb::u8).parse_next(input)?;
    let tag_list = TagList::new(tag_list.parse_next(&mut &tag_list_buffer[..])?);

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
        assert!(result.is_ok());
        if let Some(size) = result.ok() {
            assert_eq!(size, 1115);
        }
    }

    #[test]
    fn test_reading_missing_tag() {
        let tag_list = TagList::new(vec![("SOME_TAG".to_string(), "existing_value".to_string())]);
        assert!(tag_list.tag_value("MISSING_TAG").is_none());
        assert!(tag_list.parse_tag_value::<usize>("MISSING_TAG").is_err());
    }

    #[test]
    fn test_reading_int_tag() {
        let tag_list = TagList::new(vec![("SOME_TAG".to_string(), "123".to_string())]);
        assert!(tag_list.tag_value("SOME_TAG").is_some());
        assert_eq!(tag_list.tag_value("SOME_TAG").unwrap(), "123");
        assert!(tag_list.parse_tag_value::<usize>("SOME_TAG").is_ok());
        assert_eq!(tag_list.parse_tag_value::<usize>("SOME_TAG").unwrap(), 123);
    }

    #[test]
    fn test_tag_list() {
        let asset = include_bytes!("../../assets/test_32x32x8_LZ4.mlimage");
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
        let asset = include_bytes!("../../assets/test_32x32x8_LZ4.mlimage");
        let result = parse_info.parse_next(&mut &asset[..]);
        assert!(result.is_ok());
    }
}
