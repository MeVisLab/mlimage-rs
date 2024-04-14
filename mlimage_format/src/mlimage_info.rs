use ndarray::Ix;

use crate::tag_list::{TagError, TagList};

#[derive(Debug)]
pub struct MLImageInfo {
    pub endianness: winnow::binary::Endianness,
    pub dtype_size: usize,
    pub image_extent: [Ix; 6],
    pub page_extent: [Ix; 6],
    pub tag_list: TagList,
    pub uses_partial_pages: bool,
    pub world_matrix: ndarray::Array2<f64>,
}

impl MLImageInfo {
    /// Image extent in (x, y, z, c, t, u) order (logical order, first index
    /// being the fastest changing one, aka "Fortran order")
    pub fn image_extent(&self) -> [Ix; 6] {
        self.image_extent
    }

    /// Image extent in (u, c, t, z, y, x) order ("C order", memory order, with
    /// the last index being the fastest changing one)
    pub fn image_extent_c(&self) -> [Ix; 6] {
        let mut result = self.image_extent.clone();
        result.reverse();
        result
    }

    /// Page extent in (x, y, z, c, t, u) order (logical order, first index
    /// being the fastest changing one, aka "Fortran order")
    pub fn page_extent(&self) -> [Ix; 6] {
        self.image_extent
    }

    /// Page extent in (u, c, t, z, y, x) order ("C order", memory order, with
    /// the last index being the fastest changing one)
    pub fn page_extent_c(&self) -> [Ix; 6] {
        let mut result = self.page_extent.clone();
        result.reverse();
        result
    }

    pub fn page_count_per_dim(&self) -> [usize; 6] {
        let page_count_per_dim: Vec<usize> = self.image_extent
            .iter()
            .zip(self.page_extent.iter())
            .map(|(ie, pe)| num::Integer::div_ceil(ie, pe))
            .collect();
        page_count_per_dim
            .try_into()
            .expect("by construction, we must have 6D extents")
    }

    pub fn from_tag_list(tag_list: TagList) -> Result<Self, TagError> {
        let endianness = if tag_list.parse_tag_value::<u8>("ML_ENDIANESS")? > 0 {
            winnow::binary::Endianness::Big
        } else {
            winnow::binary::Endianness::Little
        };

        let dtype_size: usize = tag_list.parse_tag_value("ML_IMAGE_DTYPE_SIZE").unwrap();

        let image_extent: Vec<Ix> = "XYZCTU"
            .chars()
            .filter_map(|dim| {
                tag_list
                    .parse_tag_value(&format!("ML_IMAGE_EXT_{}", dim))
                    .ok()
            })
            .collect();
        let image_extent: [Ix; 6] = image_extent
            .try_into()
            .expect("by construction, we must have 6D extents");

        let page_extent: Vec<usize> = "XYZCTU"
            .chars()
            .filter_map(|dim| {
                tag_list
                    .parse_tag_value(&format!("ML_PAGE_EXT_{}", dim))
                    .ok()
            })
            .collect();
        let page_extent: [Ix; 6] = page_extent
            .try_into()
            .expect("by construction, we must have 6D extents");

        let uses_partial_pages = tag_list
            .parse_tag_value::<i8>("ML_USES_PARTIAL_PAGES")
            .map_or(false, |pp| pp > 0);

        let mut world_matrix = ndarray::Array2::zeros((4, 4));
        for row in 0..4 {
            for col in 0..4 {
                world_matrix[(row, col)] =
                    tag_list.parse_tag_value(&format!("ML_WORLD_MATRIX_{}{}", row, col))?;
            }
        }

        Ok(Self {
            endianness,
            dtype_size,
            image_extent,
            page_extent,
            tag_list,
            uses_partial_pages,
            world_matrix,
        })
    }
}
