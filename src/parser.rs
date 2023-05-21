use std::str::from_utf8;

use nom::{
    bytes::complete::{is_not, tag},
    character::complete as cc,
    combinator::map,
    multi::many0,
    sequence::{pair, preceded, terminated},
    IResult,
};

pub struct VersionHeader {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

pub struct MLImage {
    pub version: VersionHeader,
    pub tag_list: Vec<(String, String)>,
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
        map(is_not("\0"), |buf| from_utf8(buf).unwrap().to_owned()),
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

pub fn parse_file(input: &[u8]) -> IResult<&[u8], MLImage> {
    let (tag_list_input, version) = version_header(input)?;

    let (input, tag_list_size) = tag_list_size_in_bytes(tag_list_input)?;

    let (tag_list_input, rest) = tag_list_input.split_at(tag_list_size);
    let (nothing, tag_list) = tag_list(tag_list_input)?;

    Ok((input, MLImage { version, tag_list }))
}

#[cfg(test)]
mod tests {
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
    fn test_file() {
        let asset = include_bytes!("../assets/test_32x32x8.mlimage");
        let result = parse_file(asset);
        assert!(result.is_ok());
        if let Some((rest, image)) = result.ok() {
            assert_eq!(image.version.major, 0);
            assert_eq!(image.version.minor, 1);
        }
    }
}
