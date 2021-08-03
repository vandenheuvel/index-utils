#![feature(drain_filter)]
#![feature(is_sorted)]
#![feature(label_break_value)]

use std::cmp::Ordering;
use std::collections::HashSet;
use std::mem::swap;

#[cfg(feature = "num-traits")]
pub use num::inner_product;
#[cfg(feature = "num-traits")]
mod num;

/// Reduce the size of the vector by removing values.
///
/// # Arguments
///
/// * `vector`: `Vec` to remove indices from.
/// * `indices`: A sorted slice of indices to remove.
pub fn remove_indices<T>(vector: &mut Vec<T>, indices: &[usize]) {
    debug_assert!(indices.len() <= vector.len());
    debug_assert!(indices.is_sorted());
    // All values are unique
    debug_assert!(indices.iter().collect::<HashSet<_>>().len() == indices.len());
    debug_assert!(indices.iter().all(|&i| i < vector.len()));

    let mut i = 0;
    let mut j = 0;
    vector.retain(|_| {
        if i < indices.len() && j < indices[i] {
            j += 1;
            true
        } else if i < indices.len() { // must have j == to_remove[i]
            j += 1;
            i += 1;
            false
        } else { // i == to_remove.len()
            j += 1;
            true
        }
    });
}

/// Reduce the size of the vector by removing values.
///
/// The method operates in place.
///
/// # Arguments
///
/// * `vector`: `Vec` to remove indices from.
/// * `indices`: A sorted slice of indices to remove.
pub fn remove_sparse_indices<T>(vector: &mut Vec<(usize, T)>, indices: &[usize]) {
    debug_assert!(indices.is_sorted());
    // All values are unique
    debug_assert!(indices.iter().collect::<HashSet<_>>().len() == indices.len());

    if indices.is_empty() || vector.is_empty() {
        return;
    }

    let mut nr_skipped_before = 0;
    vector.drain_filter(|(i, _)| {
        while nr_skipped_before < indices.len() && indices[nr_skipped_before] < *i {
            nr_skipped_before += 1;
        }

        if nr_skipped_before < indices.len() && indices[nr_skipped_before] == *i {
            true
        } else {
            *i -= nr_skipped_before;
            false
        }
    });
}

/// Collect two sparse iterators by merging them.
///
/// # Arguments
///
/// * `left`: The first iterator, sorted by index.
/// * `right`: The second iterator, sorted by index.
/// * `operation`: A binary operation applied when an element is found at some index in both
/// iterators.
/// * `operation_left`: A unitary operation applied when an element is found only in the left
/// iterator.
/// * `operation_right`: A unitary operation applied when an element is found only in the right
/// iterator.
/// * `is_not_default`: A function specifying whether the result if the binary operation is the
/// default value (that is often zero).
///
/// # Return value
///
/// A vector with sparse elements by sorted and unique index.
pub fn merge_sparse_indices<I: Ord, T: Eq, U>(
    left: impl Iterator<Item=(I, T)>,
    right: impl Iterator<Item=(I, T)>,
    operation: impl Fn(T, T) -> U,
    operation_left: impl Fn(T) -> U,
    operation_right: impl Fn(T) -> U,
    is_not_default: impl Fn(&U) -> bool,
) -> Vec<(I, U)> {
    let mut results = Vec::with_capacity(left.size_hint().0.max(right.size_hint().0));

    let mut left = left.peekable();
    let mut right = right.peekable();

    while let (Some((left_index, _)), Some((right_index, _))) = (left.peek(), right.peek()) {
        let (index, result) = match left_index.cmp(&right_index) {
            Ordering::Less => {
                let (index, value) = left.next().unwrap();

                (index, operation_left(value))
            }
            Ordering::Equal => {
                let (left_index, left_value) = left.next().unwrap();
                let (_right_index, right_value) = right.next().unwrap();

                (left_index, operation(left_value, right_value))
            }
            Ordering::Greater => {
                let (index, value) = right.next().unwrap();

                (index, operation_right(value))
            }
        };

        if is_not_default(&result) {
            results.push((index, result));
        }
    }

    for (left_index, value) in left {
        results.push((left_index, operation_left(value)))
    }
    for (right_index, value) in right {
        results.push((right_index, operation_right(value)))
    }

    results
}

/// Collect two sparse iterators by merging the indices that they share, dropping the others.
///
/// # Arguments
///
/// * `left`: The first iterator, sorted by index.
/// * `right`: The second iterator, sorted by index.
/// * `operation`: A binary operation applied when an element is found at some index in both
/// iterators.
/// * `is_not_default`: A function specifying whether the result if the binary operation is the
/// default value (that is often zero).
///
/// # Return value
///
/// A vector with sparse elements by sorted and unique index.
pub fn merge_sparse_indices_intersect<I: Ord, T, U>(
    mut left: impl Iterator<Item=(I, T)>,
    mut right: impl Iterator<Item=(I, T)>,
    operation: impl Fn(T, T) -> U,
    is_not_default: impl Fn(&U) -> bool,
) -> Vec<(I, U)> {
    if let Some((mut left_index, mut left_value)) = left.next() {
        let mut results = Vec::new();

        while let Some((right_index, right_value)) = right.next() {
            match right_index.cmp(&left_index) {
                Ordering::Less => {}
                Ordering::Equal => {
                    if let Some((mut left_next_index, mut new_left_value)) = left.next() {
                        let (left_old_index, left_old_value) = {
                            swap(&mut left_index, &mut left_next_index);
                            swap(&mut left_value, &mut new_left_value);

                            (left_next_index, new_left_value)
                        };

                        let result = operation(left_old_value, right_value);
                        if is_not_default(&result) {
                            results.push((left_old_index, result));
                        }
                    } else {
                        let result = operation(left_value, right_value);
                        if is_not_default(&result) {
                            results.push((left_index, result));
                        }
                        break;
                    }
                }
                Ordering::Greater => {
                    'scope: {
                        while let Some((left_next_index, new_left_value)) = left.next() {
                            match left_next_index.cmp(&right_index) {
                                Ordering::Less => {}
                                Ordering::Equal => {
                                    let result = operation(new_left_value, right_value);
                                    if is_not_default(&result) {
                                        results.push((left_next_index, result));
                                    }
                                    if let Some((left_next_index, new_left_value)) = left.next() {
                                        left_index = left_next_index;
                                        left_value = new_left_value;
                                        break 'scope;
                                    } else {
                                        return results;
                                    }
                                }
                                Ordering::Greater => {
                                    left_index = left_next_index;
                                    left_value = new_left_value;
                                    break 'scope;
                                }
                            }
                        }

                        return results;
                    }
                }
            }
        }

        results
    } else {
        Vec::with_capacity(0)
    }
}

#[cfg(test)]
mod test {
    use std::convert::identity;
    use std::iter::empty;
    use std::ops::Add;

    use relp_num::NonZero;

    use crate::{merge_sparse_indices, merge_sparse_indices_intersect, remove_indices, remove_sparse_indices};

    #[test]
    fn test_remove_indices() {
        let mut v = vec![0f64, 1f64, 2f64];
        remove_indices(&mut v, &[1]);
        assert_eq!(v, vec![0f64, 2f64]);

        let mut v = vec![3f64, 0f64, 0f64];
        remove_indices(&mut v, &[0]);
        assert_eq!(v, vec![0f64, 0f64]);

        let mut v = vec![0f64, 0f64, 2f64, 3f64, 0f64, 5f64, 0f64, 0f64, 0f64, 9f64];
        remove_indices(&mut v, &[3, 4, 6]);
        assert_eq!(v, vec![0f64, 0f64, 2f64, 5f64, 0f64, 0f64, 9f64]);

        let mut v = vec![0f64];
        remove_indices(&mut v, &[0]);
        assert_eq!(v, vec![]);

        let mut v: Vec<i32> = vec![];
        remove_indices(&mut v, &[]);
        assert_eq!(v, vec![]);
    }

    #[test]
    fn test_remove_sparse_indices() {
        // Empty vec, removing nothing
        let mut tuples: Vec<(usize, i32)> = vec![];
        remove_sparse_indices(&mut tuples, &[]);
        assert_eq!(tuples, vec![]);

        // Non-empty vec, removing nothing
        let mut tuples = vec![(1_000, 1f64)];
        remove_sparse_indices(&mut tuples, &[]);
        assert_eq!(tuples, vec![(1_000, 1f64)]);

        // Empty vec
        let mut tuples: Vec<(usize, i32)> = vec![];
        remove_sparse_indices(&mut tuples, &[0, 1, 1_000]);
        assert_eq!(tuples, vec![]);

        // Removing value not present, index above
        let mut tuples = vec![(0, 3f64)];
        remove_sparse_indices(&mut tuples, &[1]);
        assert_eq!(tuples, vec![(0, 3f64)]);

        // Removing value not present, index below
        let mut tuples = vec![(2, 3f64)];
        remove_sparse_indices(&mut tuples, &[1]);
        assert_eq!(tuples, vec![(1, 3f64)]);

        // Removing present value, index should be adjusted
        let mut tuples = vec![(0, 0f64), (2, 2f64)];
        remove_sparse_indices(&mut tuples, &[0]);
        assert_eq!(tuples, vec![(1, 2f64)]);

        let mut tuples = vec![(0, 0), (2, 2), (4, 4)];
        remove_sparse_indices(&mut tuples, &[0, 3]);
        assert_eq!(tuples, vec![(1, 2), (2, 4)]);

        let mut tuples = vec![(0, 0), (2, 2), (4, 4)];
        remove_sparse_indices(&mut tuples, &[0, 3, 4]);
        assert_eq!(tuples, vec![(1, 2)]);
    }

    #[test]
    fn test_merge_sparse_indices() {
        // Empty
        let left: Vec<(i8, i16)> = vec![];
        let right = vec![];

        let result = merge_sparse_indices(
            left.into_iter(),
            right.into_iter(),
            Add::add,
            identity,
            identity,
            NonZero::is_not_zero,
        );
        let expected = vec![];
        assert_eq!(result, expected);

        // One empty
        let left: Vec<(i8, i16)> = vec![(2, 1)];
        let right = vec![];

        let result = merge_sparse_indices(
            left.into_iter(),
            right.into_iter(),
            Add::add,
            identity,
            identity,
            NonZero::is_not_zero,
        );
        let expected = vec![(2, 1)];
        assert_eq!(result, expected);

        // Not related
        let left = vec![(1, 6)].into_iter();
        let right = vec![(4, 9)].into_iter();

        let result = merge_sparse_indices(
            left.into_iter(),
            right.into_iter(),
            Add::add,
            identity,
            identity,
            NonZero::is_not_zero,
        );
        let expected = vec![(1, 6), (4, 9)];
        assert_eq!(result, expected);

        // Same index
        let left = vec![(1, 6)].into_iter();
        let right = vec![(1, 9)].into_iter();

        let result = merge_sparse_indices(
            left.into_iter(),
            right.into_iter(),
            Add::add,
            identity,
            identity,
            NonZero::is_not_zero,
        );
        let expected = vec![(1, 15)];
        assert_eq!(result, expected);

        // Alternating
        let left = vec![(1, 6), (3, 4)].into_iter();
        let right = vec![(2, 9)].into_iter();

        let result = merge_sparse_indices(
            left.into_iter(),
            right.into_iter(),
            Add::add,
            identity,
            identity,
            NonZero::is_not_zero,
        );
        let expected = vec![(1, 6), (2, 9), (3, 4)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_merge_sparse_indices_intersect() {
        let result: Vec<(usize, i32)> = merge_sparse_indices_intersect(
            empty(),
            empty(),
            |x: i32, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![]);

        let result = merge_sparse_indices_intersect(
            empty(),
            [(0, 1)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![]);

        let result = merge_sparse_indices_intersect(
            [(0, 1)].into_iter(),
            empty(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![]);

        let result = merge_sparse_indices_intersect(
            [(0, 1), (2, 1), (3, 1)].into_iter(),
            [(0, 1), (1, 2), (2, 4), (3, 8)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![(0, 1), (2, 4), (3, 8)]);

        let result = merge_sparse_indices_intersect(
            [(0, 1), (1, 2), (2, 4), (3, 8)].into_iter(),
            [(0, 1), (2, 1), (3, 1)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![(0, 1), (2, 4), (3, 8)]);

        let result = merge_sparse_indices_intersect(
            [(0, 1), (1, 2), (3, 8)].into_iter(),
            [(2, 1), (3, 1)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![(3, 8)]);

        let result = merge_sparse_indices_intersect(
            [(0, 1), (1, 2), (3, 8)].into_iter(),
            [(0, 1), (2, 1), (4, 1)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![(0, 1)]);

        let result = merge_sparse_indices_intersect(
            [(0, 1), (1, 2), (3, 8), (5, 7)].into_iter(),
            [(0, 1), (2, 1), (4, 1), (5, 7)].into_iter(),
            |x, y| x * y,
            |x| *x != 0,
        );
        assert_eq!(result, vec![(0, 1), (5, 49)]);
    }
}
