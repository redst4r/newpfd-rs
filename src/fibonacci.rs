//! Fibonacci encoding of integers.
//! 
//! See [here](https://en.wikipedia.org/wiki/Fibonacci_coding)
//! 
//! # Usage
//! ```rust
//! use newpfd::fibonacci::{fib_enc_multiple_fast,FibonacciDecoder, bitslice_to_fibonacci};
//! let encode = fib_enc_multiple_fast(&vec![34, 12]) ;
//! 
//! let f = FibonacciDecoder::new(&encode);
//! assert_eq!(f.collect::<Vec<_>>(), vec![34,12])
//! ```

use bitvec::prelude::*;
use itertools::izip;

/// Iterative fibonacci.
///
/// <https://github.com/rust-lang/rust-by-example>
struct Fibonacci {
    curr: u64,
    next: u64,
}

impl Iterator for Fibonacci {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        let new_next = self.curr + self.next;

        self.curr = self.next;
        self.next = new_next;

        Some(self.curr)
    }
}
/// A "constructor" for Iterative fibonacci.
#[allow(dead_code)] // only needed to generate the fibonacci sequence below
fn iterative_fibonacci() -> Fibonacci {
    Fibonacci { curr: 1, next: 1 }
}

// not sure what the significance of those settings is
// in busz, converting byte buffers to BitSlices seems to require u8;Msb01
/// The type of bitvector used in the crate.
/// Importantly, some code *relies* on `Msb0`
pub (crate) type MyBitSlice = BitSlice<u8, Msb0>;
/// reftype thqt goes with [MyBitSlice]
pub (crate) type MyBitVector = BitVec<u8, Msb0>;

// let v: Vec<_> = iterative_fibonacci().take(65 - 1).collect();
// println!("{:?}", v);
/// All fibonacci numbers up to 64bit
pub const FIB64: &[u64]= &[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 
610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 
121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578, 5702887, 
9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141, 267914296, 
433494437, 701408733, 1134903170, 1836311903, 2971215073, 4807526976, 7778742049, 
12586269025, 20365011074, 32951280099, 53316291173, 86267571272, 139583862445, 225851433717, 
365435296162, 591286729879, 956722026041, 1548008755920, 2504730781961, 4052739537881, 
6557470319842, 10610209857723, 17_167_680_177_565];
// TODO calc all fib up to u64::MAX! -> no point, we cant encode that in 64bits anyway!

/// convert a bitslice holding a single fibbonacci encoding into the numerical representation.
/// Essentially assumes that the bitslice ends with ....11 and has no other occurance of 11
pub fn bitslice_to_fibonacci(b: &MyBitSlice) -> u64{
    // omits the initial 1, i.e.
    // fib = [1,2,3,5,...]
    // let fib: Vec<_> = iterative_fibonacci().take(b.len() - 1).collect(); // shorten by one as we omit the final bit
    // println!("{:?}", fib);
    // b.ends_with(&[true, true].into());
    if b.len() > 64 {
        panic!("fib-codes cant be longer than 64bit, something is wrong!");
    }
    // TODO make sure its a proper fib-encoding (no 11 except the end)
    let mut sum = 0;
    for (bit, f) in izip!(&b[..b.len()-1], FIB64) {
        if *bit {
            sum+=f;
        }
    }
    sum
}

/// 
pub fn bitslice_to_fibonacci3(b: &MyBitSlice) -> u64{
    // omits the initial 1, i.e.
    // fib = [1,2,3,5,...]
    // let fib: Vec<_> = iterative_fibonacci().take(b.len() - 1).collect(); // shorten by one as we omit the final bit
    // println!("{:?}", fib);
    // b.ends_with(&[true, true].into());
    if b.len() > 64 {
        panic!("fib-codes cant be longer than 64bit, something is wrong!");
    }
    // TODO make sure its a proper fib-encoding (no 11 except the end)
    let mut sum = 0;
    // for (bit, f) in izip!(&b[..b.len()-1], FIB64) {
    for i in 0..b.len()-1 {
        sum+= FIB64[i]* (b[i] as u64);
    }
    sum
}

/// 
pub fn bitslice_to_fibonacci2(b: &MyBitSlice) -> u64{
    // omits the initial 1, i.e.
    // fib = [1,2,3,5,...]
    // let fib: Vec<_> = iterative_fibonacci().take(b.len() - 1).collect(); // shorten by one as we omit the final bit
    // println!("{:?}", fib);
    // b.ends_with(&[true, true].into());
    if b.len() > 64 {
        panic!("fib-codes cant be longer than 64bit, something is wrong!");
    }
    // TODO make sure its a proper fib-encoding (no 11 except the end)
    let mut sum = 0;
    for ix in b[..b.len()-1].iter_ones() {
        sum+=FIB64[ix];
    }
    sum
}

/// 
pub fn bitslice_to_fibonacci4(b: &MyBitSlice) -> u64{
    // omits the initial 1, i.e.
    // fib = [1,2,3,5,...]
    // let fib: Vec<_> = iterative_fibonacci().take(b.len() - 1).collect(); // shorten by one as we omit the final bit
    // println!("{:?}", fib);
    // b.ends_with(&[true, true].into());
    if b.len() > 64 {
        panic!("fib-codes cant be longer than 64bit, something is wrong!");
    }
    // TODO make sure its a proper fib-encoding (no 11 except the end)
    // let mut sum = 0;
    let sum = b[..b.len()-1]
        .iter()
        .by_vals()
        .enumerate()
        .filter_map(|(ix, bit)| if bit {Some(FIB64[ix])} else {None})
        .sum();
    sum
}

/// Decoder for Fibonacci encoded integer sequences
/// 
/// Constructed from a bufffer (a binary sequence) which is gradually processed
/// when iterating. The buffer remains unchanged, just the pointers into the buffer move
/// 
/// # Example
/// ```rust
/// use newpfd::fibonacci::FibonacciDecoder;
/// use bitvec::prelude::{BitVec, Msb0};
/// let buffer:BitVec<u8, Msb0> = BitVec::from_iter(vec![true, false, true, true, false, true, true]);
/// let d = FibonacciDecoder::new(buffer.as_bitslice());
/// for decoded in d {
///     println!("{}", decoded);
/// }
/// ``` 
#[derive(Debug)]
pub struct FibonacciDecoder <'a> {
    buffer: &'a MyBitSlice,
    current_pos: usize, // where we are at in the buffer (the last split), i.e the unprocessed part is buffer[current_pos..]
}

impl <'a> FibonacciDecoder<'a> {
    /// Creates a new fibonacci decoder for the given buffer. 
    /// This leaves the buffer  unchanged, just moves a pointer (self.current_pos) in the buffer around
    pub fn new(buffer: &'a MyBitSlice) -> Self {
        FibonacciDecoder { buffer, current_pos:0}
    }

    /// Returns the buffer behind the last bit processed.
    /// Comes handy when the buffer contains data OTHER than fibonacci encoded 
    /// data that needs to be processed externally.
    pub fn get_remaining_buffer(&self) -> &'a MyBitSlice{
        &self.buffer[self.current_pos..]
    }

    /// how far did we process into the buffer (pretty much the first bit after a 11).
    pub fn get_bits_processed(&self) -> usize{
        self.current_pos
    }
}

impl <'a> Iterator for FibonacciDecoder<'a> {
    type Item=u64;

    fn next(&mut self) -> Option<Self::Item> {

        let mut prev_bit = false;
        let mut accumulator = 0;
        let current_slice = &self.buffer[self.current_pos..];
        // println!("currentslice {:?}", current_slice);
        for (idx, current_bit) in current_slice.iter().by_vals().enumerate() {
            if idx > 64 {
                panic!("fib-codes cant be longer than 64bit, something is wrong!");
            }
            match (prev_bit, current_bit) {
                // current bit set, but not 11
                (false, true) => {
                    accumulator += FIB64[idx];
                }
                (true, true) => {
                    // found 11
                    let hit_len = idx+1;
                    self.current_pos += hit_len; 
                    return Some(accumulator);
                }
                (false, false) | (true, false) => {}  // current bit is zero, nothing to add
            }
            prev_bit = current_bit
        }
        None
    }
}


/// Slightly faster (2x) encoding of multiple integers into a bitvector via Fibonacci Encoding
pub fn fib_enc_multiple_fast(data: &[u64]) -> MyBitVector{

    // the capacity is a minimum, assuming each element of data is 1, i.e. `11` in fib encoding
    let mut overall = BitVec::with_capacity(2*data.len()); 

    // this just appends to the `overall` bitvec
    for &x in data {
        bits_from_table(x, FIB64, &mut overall).unwrap();
    }
    overall
}


use std::fmt::Debug;
use num::CheckedSub;
#[derive(Debug, PartialEq)]

/// Hijacked from https://github.com/antifuchs/fibonacci_codec
pub enum EncodeError<T>
where
    T: Debug + Send + Sync + 'static,
{
    /// Indicates an attempt to encode the number `0`, which can't be
    /// represented in fibonacci encoding.
    ValueTooSmall(T),

    /// A bug in fibonacci_codec in which encoding the contained
    /// number resulted in an attempt to subtract a larger fibonacci
    /// number than the number to encode.
    Underflow(T),
}
/// slightly faster fibonacci endocing (2x faster), taken from 
/// https://github.com/antifuchs/fibonacci_codec
#[inline]
pub fn bits_from_table<T>(
    n: T,
    table: &'static [T],
    result: &mut MyBitVector,
) -> Result<(), EncodeError<T>>
where
    T: CheckedSub + PartialOrd + Debug + Copy + Send + Sync + 'static,
{
    let mut current = n;
    let split_pos = table
        .iter()
        .rposition(|elt| *elt <= n)
        .ok_or(EncodeError::ValueTooSmall::<T>(n))?;

    let mut i = result.len() + split_pos + 1;
    // result.grow(split_pos + 2, false);

    result.resize(result.len() + split_pos + 2, false);
    result.set(i, true);
    for elt in table.split_at(split_pos + 1).0.iter().rev() {
        i -= 1;
        if elt <= &current {
            let next = match current.checked_sub(elt) {
                Some(next) => next,
                None => {
                    // We encountered an underflow. This is a bug, and
                    // I have no idea how it could even occur in real
                    // life. However, let's clean up and return a
                    // reasonable error:
                    result.truncate(split_pos + 2);
                    return Err(EncodeError::Underflow(n));
                }
            };
            current = next;
            result.set(i, true);
        };
    }
    Ok(())
}


#[cfg(test)]
mod test {
    use crate::fibonacci::{bitslice_to_fibonacci, FibonacciDecoder, MyBitVector, fib_enc_multiple_fast};
    use bitvec::prelude::*;

    mod test_table {
        use bitvec::vec::BitVec;
        use crate::fibonacci::{FIB64, bits_from_table, fib_enc_multiple_fast, MyBitVector};

        #[test]
        fn test_1() {
            let mut bv: MyBitVector = BitVec::new();
            bits_from_table(1, FIB64, &mut bv).unwrap();
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![true, true] 
            );
        }

        #[test]
        fn test_2() {
            let mut bv: MyBitVector = BitVec::new();
            bits_from_table(2, FIB64, &mut bv).unwrap();
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![false, true, true] 
            );
        }
        #[test]
        fn test_14() {
            let mut bv: MyBitVector = BitVec::new();
            bits_from_table(14, FIB64, &mut bv).unwrap();
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![true, false, false, false, false, true, true] 
            );
        }
        #[test]
        fn test_consecutive() {
            let mut bv: MyBitVector = BitVec::new();
            bits_from_table(1, FIB64, &mut bv).unwrap();
            bits_from_table(2, FIB64, &mut bv).unwrap();
            bits_from_table(1, FIB64, &mut bv).unwrap();
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![true, true, false,true,true, true, true] 
            );
        }

        #[test]
        fn test_fib_enc_multiple_fast() {
            let x = vec![1,2,3];
            let bv = fib_enc_multiple_fast(&x);
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![true, true, false,true,true, false, false, true, true] 
            );
        }
        
        #[test]
        fn test_fib_enc_multiple_fas_single_item() {
            let x = vec![3];
            let bv = fib_enc_multiple_fast(&x);
            assert_eq!(
                bv.iter().collect::<Vec<_>>(), 
                vec![false, false, true, true] 
            );
        }

    }

    #[test]
    fn test_fib_encode_mutiple() {
        let enc = fib_enc_multiple_fast( &vec![1,14]);
        assert_eq!(
            enc.iter().collect::<Vec<_>>(), 
            vec![true, true, true, false, false, false, false, true, true] 
        );
    }

    // #[test]
    // #[should_panic(expected = "n must be positive")]
    // fn test_fib_encode_0() {
    //     fib_enc(0);
    // }
    // #[test]
    // #[should_panic(expected = "n must be smaller than max fib")]
    // fn test_fib_encode_u64max() {
    //     fib_enc(u64::MAX);

    // }


    #[test]
    fn test_bitslice_to_fibonacci(){
        let b = bits![u8, Msb0; 1, 1];

        assert_eq!(
            bitslice_to_fibonacci(b),
            1
        );

        let b = bits![u8, Msb0; 0, 1, 1];

        assert_eq!(
            bitslice_to_fibonacci(&b),
            2
        );
        let b = bits![u8, Msb0; 0,0,1, 1];

        assert_eq!(
            bitslice_to_fibonacci(&b),
            3
        );

        let b = bits![u8, Msb0; 1,0, 1, 1];
        assert_eq!(
            bitslice_to_fibonacci(&b),
            4
        );

        let b = bits![u8, Msb0; 1,0,0,0,0,1,1];

        assert_eq!(
            bitslice_to_fibonacci(&b),
            14
        );

        let b = bits![u8, Msb0; 1,0,1,0,0,1,1];
        assert_eq!(
            bitslice_to_fibonacci(&b),
            17
        );
    }

    #[test]
    fn test_max_decode() {
        let mut v: Vec<bool> = [0_u8; 64].iter().map(|x|*x==1).collect();
        v[62]=true;
        v[63]=true;

        let b: MyBitVector  = BitVec::from_iter(v.into_iter());
        assert_eq!(
            bitslice_to_fibonacci(&b),
            10610209857723 
        );
    }

    #[test]
    fn test_myfib_decoder() {
        // let v: Vec<bool> = vec![0,0,1,1].iter().map(|x|*x==1).collect();
        // let b: MyBitVector  = BitVec::from_iter(v.into_iter());
        let b = bits![u8, Msb0; 0,0,1,1];

        // println!("full : {:?}", b);
        let mut my = FibonacciDecoder {buffer: b, current_pos:0};

        assert_eq!(
            my.next(), 
            Some(3)
        );
        assert_eq!(
            my.next(), 
            None
        );
    }

    #[test]
    fn test_myfib_decoder_consecutive_ones() {
        let b = bits![u8, Msb0; 0,0,1,1,1,1];

        println!("full : {:?}", b);
        let mut my = FibonacciDecoder {buffer: b, current_pos:0};

        assert_eq!(
            my.next(), 
            Some(3)
        );
        assert_eq!(
            my.next(), 
            Some(1)
        );
        assert_eq!(
            my.next(), 
            None
        );        
    }

    #[test]
    fn test_myfib_decoder_nothing() {
        let b = bits![u8, Msb0; 0,0,1,0,1,0,1];

        println!("full : {:?}", b);
        let mut my = FibonacciDecoder {buffer: b, current_pos:0};

        assert_eq!(
            my.next(), 
            None
        );
    }
}