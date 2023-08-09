//! Some older, slower code for fibonacci encoding
//! mostly educational
use crate::fibonacci::{FIB64, MyBitVector};

/// Fibonacci encoding of a single integer
/// 
/// # Example:
/// ```rust
/// # use newpfd::fibonacci::fib_enc;
/// let enc = fib_enc(1);  // a BitVec
/// assert_eq!(enc.iter().collect::<Vec<_>>(), vec![true, true]);
/// ```
pub fn fib_enc(mut n: u64) -> MyBitVector{

    assert!(n>0, "n must be positive");
    assert!(n<FIB64[FIB64.len()-1], "n must be smaller than max fib");

    let mut i = FIB64.len() - 1;
    let mut indices = Vec::new(); //indices of the bits that are set! will be sortet highest-lowest
    while n >0{
        // println!("n={},i={}, F[i] {}", n,i, FIB64[i] );
        if FIB64[i] <= n {
            indices.push(i);
            n -= FIB64[i];
        }
        if n == 0 { //otherwise the i-1 might cause underflow
            break
        }
        i -= 1
    }
    let max_ix = indices[0];

    let mut bits = MyBitVector::repeat(false, max_ix+1);

    // set all recoded bits
    for i in indices {
        bits.set(i, true);
    }
    // add a final 1 to get the terminator
    bits.push(true);

    bits
}

/// Encode multiple integers into a bitvector via Fibonacci Encoding
pub fn fib_enc_multiple(data: &[u64]) -> MyBitVector {
    let mut acc = MyBitVector::new();

    for &x in data {
        let mut b = fib_enc(x);
        acc.append(&mut b);
    }
    acc
}

#[cfg(test)]
mod test {
    use crate::{fibonacci::{MyBitVector}, fibonacci_old::fib_enc_multiple};
    use bitvec::prelude::*;

    use super::fib_enc;
   
    #[test]
    fn test_fib_encode_5() {
        assert_eq!(
            fib_enc(5).iter().collect::<Vec<_>>(), 
            vec![false, false, false, true, true]
         );
    }
    #[test]
    fn test_fib_encode_1() {
        assert_eq!(
            fib_enc(1).iter().collect::<Vec<_>>(), 
            vec![true, true] 
        );
    }
    #[test]
    fn test_fib_encode_14() {
        assert_eq!(
            fib_enc(14).iter().collect::<Vec<_>>(), 
            vec![true, false, false, false, false, true, true] 
        );
    }

    #[test]
    fn test_fib_encode_mutiple() {
        let enc = fib_enc_multiple( &vec![1,14]);
        assert_eq!(
            enc.iter().collect::<Vec<_>>(), 
            vec![true, true, true, false, false, false, false, true, true] 
        );
    }

    #[test]
    #[should_panic(expected = "n must be positive")]
    fn test_fib_encode_0() {
        fib_enc(0);
    }
    #[test]
    #[should_panic(expected = "n must be smaller than max fib")]
    fn test_fib_encode_u64max() {
        fib_enc(u64::MAX);
    }
}