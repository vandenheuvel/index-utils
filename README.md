[![crate](https://img.shields.io/crates/v/index-utils.svg)](https://crates.io/crates/index-utils)
[![documentation](https://docs.rs/index-utils/badge.svg)](https://docs.rs/index-utils)
[![build status](https://github.com/vandenheuvel/index-utils/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/vandenheuvel/index-utils/actions) [![codecov](https://codecov.io/gh/vandenheuvel/index-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/vandenheuvel/index-utils)

# index-utils

Utilities for working with indices, in particular with the sorted, unique index-value pairs that
make up a sparse vector.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
index-utils = "2.3.0"
```

Removing elements from a vector shifts the ones behind them, which the sparse variant accounts for:

```rust
let mut vector = vec![(0, 'a'), (2, 'b'), (4, 'c')];
index_utils::remove_sparse_indices(&mut vector, &[0, 3]);
assert_eq!(vector, vec![(1, 'b'), (2, 'c')]);
```

Two sparse vectors can be merged with `merge_sparse_indices`, or merged over only the indices that
they share with `merge_sparse_indices_intersect`. The optional `num-traits` feature adds inner
products over sparse slices and iterators.
