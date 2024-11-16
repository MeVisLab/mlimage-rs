use criterion::{criterion_group, criterion_main, Criterion};
use mlimage_rs::mlimage_format_reader::MLImageFormatReader;

fn read_large_mlimage(c: &mut Criterion) {
    let mut reader = MLImageFormatReader::open("../assets/test_large.mlimage").unwrap();
    c.bench_function("get_tile(box on 2 slices)", |b| {
        b.iter(|| {
            reader
                .get_tile::<u16>([50, 50, 1, 0, 0, 0], [50+200, 50+200, 1+2, 1, 1, 1])
                .unwrap()
        })
    });
}

fn read_part_of_large_mlimage(c: &mut Criterion) {
    let mut reader = MLImageFormatReader::open("../assets/test_large.mlimage").unwrap();
    let mut group = c.benchmark_group("slow-test");
    group.sample_size(10).bench_function("get_tile(full extent)", |b| {
        b.iter(|| {
            reader
                .get_tile::<u16>([0, 0, 0, 0, 0, 0], reader.info().image_extent())
                .unwrap()
        })
    });
}

criterion_group!(benches, read_large_mlimage, read_part_of_large_mlimage);
criterion_main!(benches);
