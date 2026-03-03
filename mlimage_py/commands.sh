(cd ../mlimage_format && cargo build --release && cargo test --release) && \
maturin-3.12 build --release && \
pip install --user --force-reinstall ../target/wheels/*whl && \
pytest test.py
