use std::sync::Arc;

use mlimage_rs::mlimage_format_reader::MLImageFormatReader;
use numpy::{ndarray::Ix, IntoPyArray, PyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tokio::sync::Mutex;

#[pyclass(name = "MLImageFormatReader")]
struct PyMLImageFormatReader {
    inner: Arc<Mutex<MLImageFormatReader>>,
}

#[pymethods]
impl PyMLImageFormatReader {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        match pyo3_async_runtimes::tokio::get_runtime().block_on(MLImageFormatReader::open(path)) {
            Ok(inner) => Ok(Self {
                inner: Arc::new(Mutex::new(inner)),
            }),
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    pub fn read_page<'py>(
        &mut self,
        py: Python<'py>,
        index: [Ix; 6],
    ) -> PyResult<Bound<'py, PyAny>> {
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals.clone(), async move {
            let mut inner = inner.lock().await;
            match inner.read_page::<u16>(index).await {
                Ok(data) => Python::attach(|py| Ok(data.into_pyarray(py).unbind())),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            }
        })
    }

    pub fn get_tile<'py>(
        &mut self,
        py: Python<'py>,
        start: [Ix; 6],
        end: [Ix; 6],
    ) -> PyResult<Bound<'py, PyAny>> {
        let locals = pyo3_async_runtimes::TaskLocals::with_running_loop(py)?.copy_context(py)?;
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py_with_locals(py, locals.clone(), async move {
            let mut inner = inner.lock().await;
            match inner.get_tile::<u16>(start, end).await {
                Ok(data) => Python::attach(|py| Ok(data.into_pyarray(py).unbind())),
                Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
            }
        })
    }

    #[getter]
    pub fn image_extent(&self) -> [Ix; 6] {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            let inner = self.inner.lock().await;
            inner.info().image_extent
        })
    }

    #[getter]
    pub fn page_extent(&self) -> [Ix; 6] {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            let inner = self.inner.lock().await;
            inner.info().page_extent
        })
    }

    #[getter]
    pub fn world_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        pyo3_async_runtimes::tokio::get_runtime().block_on(async {
            let inner = self.inner.lock().await;
            inner.info().world_matrix.clone().into_pyarray(py)
        })
    }
}

#[pymodule]
fn mlimage(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMLImageFormatReader>()?;

    Ok(())
}
