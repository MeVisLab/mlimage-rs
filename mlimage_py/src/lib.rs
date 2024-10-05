use mlimage_rs::mlimage_format_reader::MLImageFormatReader;
use numpy::{ndarray::Ix, IntoPyArray, PyArray2, PyArray6};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(name = "MLImageFormatReader")]
struct PyMLImageFormatReader {
    inner: MLImageFormatReader,
}

#[pymethods]
impl PyMLImageFormatReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        match MLImageFormatReader::open(path) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    pub fn read_page<'py>(&mut self, py: Python<'py>, index: [Ix; 6]) -> PyResult<Bound<'py, PyArray6<u16>>> {
        match self.inner.read_page(index) {
            Ok(data) => Ok(data.into_pyarray_bound(py)),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    pub fn get_tile<'py>(&mut self, py: Python<'py>, start: [Ix; 6], end: [Ix; 6]) -> PyResult<Bound<'py, PyArray6<u16>>> {
        match self.inner.get_tile(start, end) {
            Ok(data) => Ok(data.into_pyarray_bound(py)),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    #[getter]
    pub fn image_extent(&self) -> [Ix; 6] {
        self.inner.info().image_extent
    }

    #[getter]
    pub fn page_extent(&self) -> [Ix; 6] {
        self.inner.info().page_extent
    }

    #[getter]
    pub fn world_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner
            .info()
            .world_matrix
            .clone()
            .into_pyarray_bound(py)
    }
}

#[pymodule]
fn mlimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMLImageFormatReader>()?;

    Ok(())
}
