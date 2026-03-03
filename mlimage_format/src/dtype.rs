#[derive(Debug, PartialEq)]
pub enum DType {
    MLint8,
    MLuint8,
    MLint16,
    MLuint16,
    MLint32,
    MLuint32,
    MLint64,
    MLuint64,
    MLfloat,
    MLdouble,
}

impl DType {
    pub fn name(&self) -> &'static str {
        match self {
            DType::MLint8 => "int8",
            DType::MLuint8 => "unsigned int8",
            DType::MLint16 => "int16",
            DType::MLuint16 => "unsigned int16",
            DType::MLint32 => "int32",
            DType::MLuint32 => "unsigned int32",
            DType::MLint64 => "int64",
            DType::MLuint64 => "unsigned int64",
            DType::MLfloat => "float",
            DType::MLdouble => "double",
        }
    }

    pub fn new_from_name(name: &str) -> Option<Self> {
        match name {
            "int8" => Some(DType::MLint8),
            "unsigned int8" => Some(DType::MLuint8),
            "int16" => Some(DType::MLint16),
            "unsigned int16" => Some(DType::MLuint16),
            "int32" => Some(DType::MLint32),
            "unsigned int32" => Some(DType::MLuint32),
            "int64" => Some(DType::MLint64),
            "unsigned int64" => Some(DType::MLuint64),
            "float" => Some(DType::MLfloat),
            "double" => Some(DType::MLdouble),
            _ => None,
        }
    }
}

pub trait GetMLDType {
    const ML_DTYPE: DType;
}

macro_rules! impl_get_ml_dtype {
    ($rust_type:ty, $dtype_variant:ident) => {
        impl GetMLDType for $rust_type {
            const ML_DTYPE: DType = {
                DType::$dtype_variant
            };
        }
    };
}

impl_get_ml_dtype!(u8, MLuint8);
impl_get_ml_dtype!(i8, MLint8);
impl_get_ml_dtype!(u16, MLuint16);
impl_get_ml_dtype!(i16, MLint16);
impl_get_ml_dtype!(u32, MLuint32);
impl_get_ml_dtype!(i32, MLint32);
impl_get_ml_dtype!(u64, MLuint64);
impl_get_ml_dtype!(i64, MLint64);
impl_get_ml_dtype!(f32, MLfloat);
impl_get_ml_dtype!(f64, MLdouble);
