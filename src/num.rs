use std::ops::{Mul, AddAssign};
use std::cmp::Ordering;
use std::mem::swap;

/// Compute the inner product of two sparse iterators.
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
pub fn inner_product<I: Ord, T1, T2, O>(
    mut left: impl Iterator<Item=(I, T1)>,
    mut right: impl Iterator<Item=(I, T2)>,
) -> O
where
    T1: Mul<T2, Output=O>,
    O: num_traits::Zero + AddAssign,
{
    let mut total = O::zero();

    if let Some((mut left_index, mut left_value)) = left.next() {
        while let Some((right_index, right_value)) = right.next() {
            match right_index.cmp(&left_index) {
                Ordering::Less => {}
                Ordering::Equal => {
                    if let Some((left_next_index, mut new_left_value)) = left.next() {
                        left_index = left_next_index;
                        let left_old_value = {
                            swap(&mut left_value, &mut new_left_value);
                            new_left_value
                        };

                        total += left_old_value * right_value;
                    } else {
                        total += left_value * right_value;
                        break;
                    }
                }
                Ordering::Greater => {
                    'scope: {
                        while let Some((left_next_index, new_left_value)) = left.next() {
                            match left_next_index.cmp(&right_index) {
                                Ordering::Less => {}
                                Ordering::Equal => {
                                    total += new_left_value * right_value;
                                    if let Some((left_next_index, new_left_value)) = left.next() {
                                        left_index = left_next_index;
                                        left_value = new_left_value;
                                        break 'scope;
                                    } else {
                                        return total;
                                    }
                                }
                                Ordering::Greater => {
                                    left_index = left_next_index;
                                    left_value = new_left_value;
                                    break 'scope;
                                }
                            }
                        }

                        return total;
                    }
                }
            }
        }
    }

    total
}

pub fn inner_product_slice_iter<'a, 'b, I: Ord, T1, T2: 'b, J: Iterator<Item=(I, &'b T2)>, O>(
    left: &'a [(I, T1)], right: J,
) -> O
where
    &'a T1: Mul<&'b T2, Output=O>,
    O: num_traits::Zero + AddAssign,
{
    let mut total = O::zero();

    let mut i = 0;
    for (index, value) in right {
        while i < left.len() && left[i].0 < index {
            i += 1;
        }

        if i < left.len() && left[i].0 == index {
            total += &left[i].1 * value;
            i += 1;
        }

        if i == left.len() {
            break;
        }
    }

    total
}

/// Calculate the inner product between two vectors.
///
/// # Arguments
///
/// * `left`: The first slice, sorted by index.
/// * `right`: The second slice, sorted by index.
///
/// # Return value
///
/// The inner product.
#[must_use]
pub fn inner_product_slice<'a, I: Ord, T1, T2, O>(left: &'a [(I, T1)], right: &'a [(I, T2)]) -> O
where
    &'a T1: Mul<&'a T2, Output=O>,
    O: num_traits::Zero + AddAssign,
{
    debug_assert!(left.is_sorted_by_key(|(i, _)| i));
    debug_assert!(left.windows(2).all(|w| w[0].0 < w[1].0));
    debug_assert!(right.is_sorted_by_key(|(i, _)| i));
    debug_assert!(right.windows(2).all(|w| w[0].0 < w[1].0));

    let mut total = O::zero();

    let mut left_lowest = 0;
    let mut right_lowest = 0;

    while left_lowest < left.len() && right_lowest < right.len() {
        let self_sought = &left[left_lowest].0;
        let other_sought = &right[right_lowest].0;
        match self_sought.cmp(other_sought) {
            Ordering::Less => {
                match left[left_lowest..].binary_search_by_key(&other_sought, |(i, _)| i) {
                    Err(diff) => {
                        left_lowest += diff;
                        right_lowest += 1;
                    },
                    Ok(diff) => {
                        total += &left[left_lowest + diff].1 * &right[right_lowest].1;
                        left_lowest += diff + 1;
                        right_lowest += 1;
                    },
                }
            },
            Ordering::Greater => {
                match right[right_lowest..].binary_search_by_key(&self_sought, |(i, _)| i) {
                    Err(diff) => {
                        left_lowest += 1;
                        right_lowest += diff;
                    },
                    Ok(diff) => {
                        total += &left[left_lowest].1 * &right[right_lowest + diff].1;
                        left_lowest += 1;
                        right_lowest += diff + 1;
                    },
                }
            },
            Ordering::Equal => {
                total += &left[left_lowest].1 * &right[right_lowest].1;
                left_lowest += 1;
                right_lowest += 1;
            },
        }
    }

    total
}

#[cfg(test)]
mod test {
    use crate::num::{inner_product, inner_product_slice};
    use std::iter::empty;

    #[test]
    fn test_inner_product() {
        let result: i32 = inner_product(
            empty::<(usize, i32)>(),
            empty::<(_, i32)>(),
        );
        assert_eq!(result, 0);

        let result = inner_product(
            empty::<(usize, i32)>(),
            [(0, 1)].into_iter(),
        );
        assert_eq!(result, 0);

        let result = inner_product(
            empty::<(usize, &i32)>(),
            [(0, 1)].into_iter(),
        );
        assert_eq!(result, 0);

        let result: i32 = inner_product(
            [(0, 1)].into_iter(),
            empty::<(_, i32)>(),
        );
        assert_eq!(result, 0);

        let result = inner_product(
            [(0, 1), (2, 1), (3, 1)].into_iter(),
            [(0, 1), (1, 2), (2, 4), (3, 8)].into_iter(),
        );
        assert_eq!(result, 1 + 4 + 8);

        let result = inner_product(
            [(0, 1), (1, 2), (2, 4), (3, 8)].into_iter(),
            [(0, 1), (2, 1), (3, 1)].into_iter(),
        );
        assert_eq!(result, 1 + 4 + 8);

        let result = inner_product(
            [(0, 1), (1, 2), (3, 8)].into_iter(),
            [(2, 1), (3, 1)].into_iter(),
        );
        assert_eq!(result, 8);

        let result = inner_product(
            [(0, 1), (1, 2), (3, 8)].into_iter(),
            [(0, 1), (2, 1), (4, 1)].into_iter(),
        );
        assert_eq!(result, 1);

        assert_eq!(
            inner_product(
                [(0, 1), (1, 2), (3, 8), (5, 7)].into_iter(),
                [(0, 1), (2, 1), (4, 1), (5, 7)].into_iter(),
            ),
            50,
        );

        assert_eq!(
            inner_product(
                [(0, 5), (2, 7)].into_iter(),
                [(1, 2)].into_iter(),
            ),
            0,
        );

        assert_eq!(
            inner_product(
                [(1, 2)].into_iter(),
                [(0, 5), (2, 7)].into_iter(),
            ),
            0,
        );

        assert_eq!(
            inner_product(
                [(0, 1), (3, 1), (12, 1), (13, 1)].into_iter(),
                [(0, -1), (1, 1), (2, 1), (12, 1)].into_iter(),
            ),
            0,
        );
    }

    #[test]
    fn test_inner_product_slice() {
        assert_eq!(inner_product_slice::<usize, i32, i32, _>(&[], &[]), 0);
        assert_eq!(inner_product_slice::<i32, i32, i32, _>(&[], &[(1, 5), (2, 7)]), 0);
        assert_eq!(inner_product_slice::<_, _, i32, _>(&[(1, 5), (2, 7)], &[]), 0);
        assert_eq!(inner_product_slice(&[(0, 3)], &[(0, 5)]), 3 * 5);
        assert_eq!(inner_product_slice(&[(1, 3)], &[(1, 5)]), 3 * 5);
        assert_eq!(inner_product_slice(&[(0, 3)], &[(1, 5)]), 0);
        assert_eq!(inner_product_slice(&[(1, 3)], &[(0, 5)]), 0);
        assert_eq!(inner_product_slice(&[(1, 5), (2, 6)], &[(1, 5), (2, 6)]), 5 * 5 + 6 * 6);
        assert_eq!(inner_product_slice(&[(1, 2), (2, 3)], &[(1, 5), (2, 7)]), 2 * 5 + 3 * 7);
        assert_eq!(inner_product_slice(&[(2, 3)], &[(0, 5), (1, 7)]), 0);
        assert_eq!(inner_product_slice(&[(0, 3)], &[(0, 5), (1, 7)]), 3 * 5);
        assert_eq!(inner_product_slice(&[(1, 3)], &[(0, 5), (1, 7)]), 3 * 7);
        assert_eq!(inner_product_slice(&[(1, 3)], &[(0, 5), (2, 7)]), 0);
        assert_eq!(
            inner_product_slice(
            &[(0, 1), (3, 1), (13, 1), (14, 1)],
            &[(0, -1), (1, 1), (2, 1), (13, 1)],
            ),
            0,
        );
    }
}
