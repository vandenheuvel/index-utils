use std::borrow::Borrow;
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
pub fn inner_product<I: Ord, S, T: Borrow<S>, O>(
    mut left: impl Iterator<Item=(I, T)>,
    mut right: impl Iterator<Item=(I, T)>,
) -> O
where
    for<'r> &'r S: Mul<Output=O>,
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

                        total += left_old_value.borrow() * right_value.borrow();
                    } else {
                        total += left_value.borrow() * right_value.borrow();
                        break;
                    }
                }
                Ordering::Greater => {
                    'scope: {
                        while let Some((left_next_index, new_left_value)) = left.next() {
                            match left_next_index.cmp(&right_index) {
                                Ordering::Less => {}
                                Ordering::Equal => {
                                    total += new_left_value.borrow() * right_value.borrow();
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

#[cfg(test)]
mod test {
    use crate::num::inner_product;
    use std::iter::empty;

    #[test]
    fn test_inner_product() {
        let result: i32 = inner_product(
            empty::<(usize, i32)>(),
            empty(),
        );
        assert_eq!(result, 0);

        let result = inner_product(
            empty::<(usize, i32)>(),
            [(0, 1)].into_iter(),
        );
        assert_eq!(result, 0);

        let result = inner_product(
            [(0, 1)].into_iter(),
            empty(),
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
}
