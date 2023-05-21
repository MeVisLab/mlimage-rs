use std::str::from_utf8;

use nom::{
    bytes::complete::{is_not, tag},
    character::complete as cc,
    combinator::map,
    sequence::{pair, terminated},
    IResult,
};

pub struct VersionHeader {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

pub fn version_header(input: &[u8]) -> IResult<&[u8], VersionHeader> {
    let (input, _header) = tag("MLImageFormatVersion.")(input)?;
    let (input, major) = terminated(cc::u16, cc::char('.'))(input)?;
    let (input, minor) = terminated(cc::u16, cc::char('.'))(input)?;
    let (input, patch) = terminated(cc::u16, cc::char('\0'))(input)?;
    Ok((input, VersionHeader{ major, minor, patch }))
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
}
