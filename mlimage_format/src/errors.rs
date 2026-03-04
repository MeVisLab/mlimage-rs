use std::fmt::Display;

use winnow::error::ContextError;

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

#[derive(Debug, Default)]
pub struct CompressionError {
    msg: String,
}

impl std::error::Error for CompressionError {}

impl CompressionError {
    pub fn new(msg: String) -> Self {
        Self { msg }
    }
}

impl Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compression error: {}", self.msg)
    }
}
