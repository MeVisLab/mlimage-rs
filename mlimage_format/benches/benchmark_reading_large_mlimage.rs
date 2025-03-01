use criterion::{criterion_group, criterion_main, Criterion};
use mlimage_rs::mlimage_format_reader::MLImageFormatReader;

fn read_part_of_large_mlimage(c: &mut Criterion) {
    c.bench_function("get_tile(box on 2 slices)", |b| {
        let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();
        b.to_async(&rt).iter(move || async {
            let mut reader = MLImageFormatReader::open("../assets/test_large.mlimage")
                .await
                .unwrap();
            reader
                .get_tile::<u16>([50, 50, 1, 0, 0, 0], [50 + 200, 50 + 200, 1 + 2, 1, 1, 1])
                .await
                .unwrap()
        })
    });
}

fn read_large_mlimage(c: &mut Criterion) {
    let mut group = c.benchmark_group("slow-test");
    group
        .sample_size(10)
        .bench_function("get_tile(full extent)", |b| {
            let rt = tokio::runtime::Builder::new_multi_thread().build().unwrap();
            b.to_async(&rt).iter(|| async {
                let mut reader = MLImageFormatReader::open("../assets/test_large.mlimage")
                    .await
                    .unwrap();
                reader
                    .get_tile::<u16>([0, 0, 0, 0, 0, 0], reader.info().image_extent())
                    .await
                    .unwrap()
            })
        });
}

criterion_group!(benches, read_part_of_large_mlimage, read_large_mlimage);
criterion_main!(benches);
