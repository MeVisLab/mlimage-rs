use std::{fmt::Display, str::FromStr};

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
    pub fn new(tags: Vec<(String, String)>) -> Self {
        Self(tags)
    }

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
