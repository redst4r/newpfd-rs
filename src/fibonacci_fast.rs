//! Fast Fibonacci Encoding Algorithm,
//! see this [paper](https://www.researchgate.net/publication/220827231_Fast_Fibonacci_Encoding_Algorithm)
//! 
//! Basically, instead of decoding bit by bit, we do larger segments of bits, where
//! we precomputed their decoded representation already in a lookup table.
//! 
//! The tricky part: A segment might have a encoded number, but also parts of the next encoded number:
//! ```bash,no_run  
//! segment1|segment 2
//! 00001100|11000011
//!       ----- Remainder 
//! ```
//! 
//! Turns out that the implementation details are pretty important. 
//! Implementing a simple lookup table as a HashMap<Segment, ...> is actually **slower** than simple bit-by-bit decoding.
//! One has to exploit the fact that a segment can be represented by an integer,
//! and store the results in a vec, indexed by the segment

use std::{collections::HashMap, hash::Hash};
use bitvec::{vec::BitVec, field::BitField, view::BitView};
use itertools::Itertools;
use num::traits::Pow;
use bitvec::prelude::*;
use crate::{fibonacci::FIB64, fib_utils::fibonacci_left_shift};
use std::cmp;


/// used to remember the position in the bitvector and the partically decoded number
#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
pub struct State(pub usize);

/// Result of the finite state machine in the paper
/// for a given (state, input segment), it yields the next state
/// and this result structure containing decoded numbers and some info about
/// the trailing bits not yet decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodingResult {
    numbers: Vec<u64>, // fully decoded numbers from the segment
    /// Stores the partially decoded number left in hte segment
    u: u64,  // partially decoded number; i.e. anything that didnt have a delimiter so far, kind of the accumulator
    lu: usize,  // bitlength of U; i..e how many bits processed so far without hitting the delimieter
}

// lookup using the bitvec as an action on the statemachine
fn create_lookup(segment_size: usize) -> HashMap<(State, BitVec<usize, Lsb0>), (State, DecodingResult)> {
    
    let mut table: HashMap<(State, BitVec<usize, Lsb0>), (State, DecodingResult)> = HashMap::new();

    for lastbit in [true, false]{
        for s in 0..2.pow(segment_size) {
            let bitstream = (s as usize).view_bits_mut::<Lsb0>()[..segment_size].to_owned();
            
            // determining the new state is a bit more tricky than just looking
            // at the last bit of the segment:
            // 1. if the segment terminated with (11), this resets the state to 0
            //     xxxxxx11|1
            //     otherwise this would immediately terminate in [0]. But really the last bit
            //     has been used up in the terminator
            // 2. otherwise, we still have to be carefull:
            //     00000111|10000000
            //          OR
            //     000(11)(11)1|10000000
            //    Even though the last bits are 11, this is NOT a terminator
            //    and the next state indeed is 1
            //
            // The easiest way is to just see what comes back from the decoding:
            // We get the number of bits after the last terminator and can hence
            // pull out all the bits in questions

            let r = decode_with_remainder(&bitstream, lastbit);
            let newstate;
            // we need to know the bits behind the last terminator
            let trailing_bits= &bitstream[bitstream.len()-r.lu..];
            if trailing_bits.len() == 0 {
                // we ended with a temrinator:  xxxxxx11|yyy
                // resets the state to 0
                newstate = State(0);
            } else {
                let final_bit = trailing_bits[trailing_bits.len()-1];
                newstate = if final_bit { State(1)} else {State(0)}
            }

            let laststate = if lastbit {State(1)} else {State(0)};
            table.insert((laststate, bitstream), (newstate, r));
        }
    }
    table
}

/// decodes a fibonacci stream until the very end of the stream
/// there might be a remainder (behind the last 11 delimiter)
/// which is also returned (its value in Fib, and its len in fib) 
fn decode_with_remainder(bitstream: &BitSlice<usize, Lsb0>, lastbit_external: bool) -> DecodingResult{

    assert!(bitstream.len() < 64, "fib-codes cant be longer than 64bit, something is wrong!");

    let mut lastbit = lastbit_external;
    let mut decoded_ints = Vec::new();
    let mut accum = 0_u64;
    let mut offset = 0;
    for b in bitstream.iter().by_vals() {
        match (lastbit, b) {
            (false, true) => {
                accum+= FIB64[offset];
                offset+=1;
                lastbit = b;
            }
            (true, true) => {
                // found delimiter; Note, the bit has already been counted in acc
                decoded_ints.push(accum);

                // reset for next number
                accum = 0;
                offset = 0;
                // BIG TRICK: we need to set lastbit=false for the next iteration; otherwise 011|11 picks up the next 11 as delimiter
                lastbit=false;
            }
            (false, false) | (true, false) => {
                // nothing to add, jsut increase offset
                offset+=1;
                lastbit = b;
            }
        }
    }
    DecodingResult {numbers: decoded_ints, u: accum, lu: offset}
}

#[cfg(test)]
mod test_decode_with_remainder{
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};
    #[test]
    fn test_decode_with_remainder_edge_case_delimiters() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![usize, Lsb0; 0,1,1,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(r.numbers, vec![2,1]);
        assert_eq!(r.u, 0);
        assert_eq!(r.lu, 0);
    }

    #[test]
    fn test_decode_with_remainder_edge_case_delimiters2() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![usize, Lsb0; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
        let r = decode_with_remainder(&bits, false);
        assert_eq!(r.numbers, vec![21,22]);
        assert_eq!(r.u, 0);
        assert_eq!(r.lu, 0);
    }

    // #[test]
    fn test_decode_with_remainder_from_table94() {

        // the exampel from the paper, Fig 9.4
        let mut bits = &181.view_bits::<Lsb0>()[..8];
        // bits.reverse();
        
        println!("{bits:?}");
        // let bits = bits![usize, Lsb0; 0,1,1,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(r.numbers, vec![4]);
        assert_eq!(r.u, 7);
        assert_eq!(r.lu, 4);

        // the exampel from the paper, Fig 9.4
        let bits = &165_usize.view_bits::<Lsb0>()[..8];
        println!("{:?}", bits);
        let r = decode_with_remainder(&bits, true);
        assert_eq!(r.numbers, vec![0]);
        assert_eq!(r.u, 31);
        assert_eq!(r.lu, 7);

        // the exampel from the paper, Fig 9.4
        let bits = &114_usize.view_bits::<Lsb0>()[..8];
        let r = decode_with_remainder(&bits, true);
        assert_eq!(r.numbers, vec![2]);
        assert_eq!(r.u, 6);
        assert_eq!(r.lu, 5);            
    }


    #[test]
    fn test_decode_with_remainder() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(r.numbers, vec![4,7]);
        assert_eq!(r.u, 31);
        assert_eq!(r.lu, 7);

        // the exampel from the paper, Fig 9.4
        // no remainder this time
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(r.numbers, vec![4,7]);
        assert_eq!(r.u, 0);
        assert_eq!(r.lu, 0);    
    }
}
/// 
pub fn fast_decode(stream: BitVec<usize, Lsb0>, segment_size: usize) -> Vec<u64> {

    let the_table = create_lookup(segment_size);

    let mut decoded_numbers = Vec::new();
    let mut n_chunks = stream.len() / segment_size;
    if stream.len() % segment_size > 0 {
        n_chunks+= 1;
    } 

    let mut state = State(0);
    let mut len = 0;
    let mut n = 0_u64;
    // for segment in &stream.iter().chunks(8) {
    for i in 0..n_chunks{
        // get a segment
        let start = i*segment_size;
        let end = cmp::min((i+1)*segment_size, stream.len());
        let mut segment = stream[start..end].to_bitvec();
        // // the last segment might be shorter, we need to pad it
        for _ in 0..segment_size-segment.len() {
            segment.push(false);
        }
        
        // println!("{state:?} {segment:?}", );
        // state machine
        if !the_table.contains_key(&(state, segment.clone())) {
            println!("{state:?}, {segment:?}");
        }

        let (newstate, result) = the_table.get(&(state, segment)).unwrap();

        state = *newstate;
        let mutresult = result;
        // 
        for &num in mutresult.numbers.iter() {
            if len > 0 {  // some leftover from the previous segment
                n += fibonacci_left_shift(num, len);
            } else {
                n = num;
            }

            decoded_numbers.push(n);
            n = 0;
            len = 0;
        }

        // the beginning and inner port of F(n)
        if result.lu > 0 {
            n += fibonacci_left_shift(result.u, len);
            len += result.lu;
        }
    }
    decoded_numbers
}

/// operating on 8bit seqgments
/// we explicityl chop the bitstream into u8-segments
/// -> each segment can be represented by a number, which solves the problem of slow hashmap access
pub fn fast_decode_u8(stream: BitVec<usize, Lsb0>, table: &impl U8Lookup) -> Vec<u64> {

    let segment_size = 8;

    let mut decoded_numbers = Vec::new();
    // let mut n_chunks = stream.len() / segment_size;
    // if stream.len() % segment_size > 0 {
    //     n_chunks+= 1;
    // } 

    let mut state = State(0);
    let mut len = 0;
    let mut n = 0_u64;

    // for i in 0..n_chunks{
    for segment in stream.chunks(segment_size) {

        // // get a segment
        // let start = i*segment_size;
        // let end = cmp::min((i+1)*segment_size, stream.len());
        // let segment = &stream[start..end];
        let segment_u8 = segment.load_be::<u8>();
        
        let (newstate, result) = table.lookup(state, segment_u8);
        state = newstate;

        for &num in result.numbers.iter() {
            if len > 0 {  // some leftover from the previous segment
                n += fibonacci_left_shift(num, len);
            } else {
                n = num;
            }

            decoded_numbers.push(n);
            n = 0;
            len = 0;
        }

        // the beginning and inner port of F(n)
        if result.lu > 0 {
            n += fibonacci_left_shift(result.u, len);
            len += result.lu;
        }
    }
    decoded_numbers
}

/// operating on 16bit seqgments
/// we explicityl chop the bitstream into u8-segments
/// -> each segment can be represented by a number, which solves the problem of slow hashmap access
pub fn fast_decode_u16(stream: BitVec<usize, Lsb0>, table: &impl U16Lookup) -> Vec<u64> {

    // println!("Total stream {}", bitstream_to_string(&stream));
    let segment_size = 16;

    let mut decoded_numbers = Vec::new();
    // let mut n_chunks = stream.len() / segment_size;
    // if stream.len() % segment_size > 0 {
    //     n_chunks+= 1;
    // } 

    let mut state = State(0);
    let mut len = 0;
    let mut n = 0_u64;

    // for i in 0..n_chunks{
    for segment in stream.chunks(segment_size) {

        // get a segment
        // let start = i*segment_size;
        // let end = cmp::min((i+1)*segment_size, stream.len());
        // let segment = &stream[start..end];

        let segment_u16 = segment.load_be::<u16>();
        let (newstate, result) = table.lookup(state, segment_u16);

        state = newstate;

        for &num in result.numbers.iter() {
            if len > 0 {  // some leftover from the previous segment
                n += fibonacci_left_shift(num, len);
            }
            else {
                n = num;
            }

            decoded_numbers.push(n);
            n= 0;
            len =0;
        }

        // the beginning and inner port of F(n)
        if result.lu > 0 {
            n += fibonacci_left_shift(result.u, len);
            len += result.lu;
        }
    }
    decoded_numbers
}

#[cfg(test)]
mod testing_fast_decode {
    use super::*;
    use pretty_assertions::assert_eq;
    #[test]
    fn test_fast_decode() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0].to_bitvec();

        let r = fast_decode(bits.clone(), 8);
        assert_eq!(r, vec![4,7, 86]);

        let table = LookupU8Vec::new();
        let r = fast_decode_u8(bits.clone(), &table);
        assert_eq!(r, vec![4,7, 86]);

        let table = LookupU16Vec::new();
        let r = fast_decode_u16(bits, &table);
        assert_eq!(r, vec![4,7, 86]);    
    }

    #[test]
    fn test_fast_decode_111_at_segment_border() {
        // edge case when the delimitator is algined with the segment and the next segment starts with 1
        // make sure to no double count 
        let bits = bits![usize, Lsb0; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
        let r = fast_decode(bits.clone(), 8);
        assert_eq!(r, vec![21, 22]);

        let table = LookupU8Vec::new();
        let r = fast_decode_u8(bits.clone(), &table);
        assert_eq!(r, vec![21, 22]);    

        let r = fast_decode_u8(bits, &table);
        assert_eq!(r, vec![21, 22]);      
    }
    #[test]
    fn test_fast_decode_special_case() {
        // edge case when theres a bunch of 1111 at the end of the segment
        // we need to make sure that we dervie the new state correctly

        let bits = bits![usize, Lsb0; 
            0,1,1,1,0,1,1,1,
            1].to_bitvec();
        let expected = vec![2, 4, 1];
        
        let r = fast_decode(bits.clone(), 8);
        assert_eq!(r, expected);

        let table = LookupU8Vec::new();
        let r = fast_decode_u8(bits.clone(), &table);
        assert_eq!(r, expected);    

        let table = LookupU8Hash::new();
        let r = fast_decode_u8(bits.clone(), &table);
        assert_eq!(r, expected);   

        // fr the u16, we need a much longer segment
        let bits = bits![usize, Lsb0; 
            1,1,1,1,1,1,1,1,
            0,1,1,1,0,1,1,1,
            1].to_bitvec();
        let expected = vec![1,1,1,1, 2, 4, 1];

        let table = LookupU16Vec::new();
        let r = fast_decode_u16(bits.clone(), &table);
        assert_eq!(r, expected);      

        let table = LookupU16Hash::new();
        let r = fast_decode_u16(bits, &table);
        assert_eq!(r, expected);      
    }


    #[test]
    fn test_correctness_fast_decode() {
        use crate::fibonacci::FibonacciDecoder;
        use crate::fib_utils::random_fibonacci_stream;
        // use crate::newpfd_bitvec::bitstream_to_string;
        let b = random_fibonacci_stream(100000, 1, 1000);
        // let b = dummy_encode(vec![64, 11, 88]);
        // make a copy for fast decoder
        let mut b_fast: BitVec<usize, Lsb0> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b);
        let x1: Vec<_> = dec.collect();
        let x2 = fast_decode(b_fast.clone(), 8);

        println!("{:?}", bitstream_to_string(&b_fast));

        assert_eq!(x1, x2);
    }

    #[test]
    fn test_correctness_fast_decode_u8() {
        use crate::fibonacci::FibonacciDecoder;
        use crate::fib_utils::random_fibonacci_stream;
        let b = random_fibonacci_stream(100000, 1, 1000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<usize, Lsb0> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b);
        let x1: Vec<_> = dec.collect();

        let table = LookupU8Vec::new();
        let x2 = fast_decode_u8(b_fast.clone(), &table);
        assert_eq!(x1, x2);

        let table = LookupU8Hash::new();
        let x2 = fast_decode_u8(b_fast.clone(), &table);
        assert_eq!(x1, x2);
    }

    #[test]
    fn test_correctness_fast_decode_u16() {
        use crate::fibonacci::FibonacciDecoder;
        use crate::fib_utils::random_fibonacci_stream;
        let b = random_fibonacci_stream(100000, 1, 1000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<usize, Lsb0> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b);
        let x1: Vec<_> = dec.collect();
        let table = LookupU16Vec::new();
        let x2 = fast_decode_u16(b_fast.clone(), &table);
        assert_eq!(x1, x2);

        let table = LookupU16Hash::new();
        let x2 = fast_decode_u16(b_fast.clone(), &table);
        assert_eq!(x1, x2);
    }

    // #[test]
    #[allow(dead_code)]
    fn test_fast_speed() {
        use crate::fib_utils::random_fibonacci_stream;

        let b = random_fibonacci_stream(1000000, 100000, 1000000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<usize, Lsb0> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }    
        // let x2 = fast_decode(b_fast.clone(), 8);
        let table = LookupU8Vec::new();
        let x2 = fast_decode_u8(b_fast.clone(), &table);

        println!("{}", x2.iter().sum::<u64>());

        // let x2 = fast_decode(b_fast.clone(), 8);
        let table = LookupU16Vec::new();
        let x2 = fast_decode_u16(b_fast.clone(), &table);
        println!("{}", x2.iter().sum::<u64>())    

    }
}
/// just for debugging purpose
pub fn bitstream_to_string<T:BitStore>(buffer: &BitSlice<T, Lsb0>) -> String{
    let s = buffer.iter().map(|x| if *x{"1"} else {"0"}).join("");
    s
}

/// Fast Fibonacci decoding lookup table for 8bit segments
pub trait U8Lookup {
    /// given the state of the last decoding operation and the new segment, returns
    /// the (precomputed) new state and decoding result
    fn lookup(&self, s: State, segment: u8) -> (State, &DecodingResult);
}

/// Vector based lookup table for u8
pub struct LookupU8Vec {
    table_state0: Vec<(State, DecodingResult)>,
    table_state1: Vec<(State, DecodingResult)>,
}
impl LookupU8Vec {
    /// create a new Lookup table for fast fibonacci decoding using 8bit segments
    /// This implementation uses a Vec
    pub fn new() -> Self {
        let segment_size = 8;

        let mut table_state0 = Vec::new();
        let mut table_state1 = Vec::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let bitstream = (s as usize).view_bits_mut::<Lsb0>()[..segment_size].to_owned();
                
                // determining the new state is a bit more tricky than just looking
                // at the last bit of the segment:
                // 1. if the segment terminated with (11), this resets the state to 0
                //     xxxxxx11|1
                //     otherwise this would immediately terminate in [0]. But really the last bit
                //     has been used up in the terminator
                // 2. otherwise, we still have to be carefull:
                //     00000111|10000000
                //          OR
                //     000(11)(11)1|10000000
                //    Even though the last bits are 11, this is NOT a terminator
                //    and the next state indeed is 1
                //
                // The easiest way is to just see what comes back from the decoding:
                // We get the number of bits after the last terminator and can hence
                // pull out all the bits in questions

                let r = decode_with_remainder(&bitstream, lastbit);
                let newstate;
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                if trailing_bits.len() == 0 {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    newstate = State(0);
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    newstate = if final_bit { State(1)} else {State(0)}
                }

                // insert result based on new state
                if lastbit {
                    table_state1.push((newstate, r))
                }
                else{
                    table_state0.push((newstate, r))
                }
            }
        }
        LookupU8Vec { table_state0, table_state1}
    }
}

impl U8Lookup for LookupU8Vec {
    fn lookup(&self, s: State, segment: u8) -> (State, &DecodingResult) {
        let (newstate, result) = match s {
            State(0) => self.table_state0.get(segment as usize).unwrap(),
            State(1) => self.table_state1.get(segment as usize).unwrap(),
            State(_) => panic!("yyy")
        };
        (*newstate, result)
    }
}

/// HashMap based lookup table for u8
pub struct LookupU8Hash {
    table_state0: HashMap<u8, (State, DecodingResult)>,
    table_state1: HashMap<u8, (State, DecodingResult)>,
}
impl LookupU8Hash {
    /// create a new Lookup table for fast fibonacci decoding using 8bit segments
    /// This implementation uses a HashMap
    pub fn new() -> Self {
        let segment_size = 8;

        let mut table_state0 = HashMap::new();
        let mut table_state1 = HashMap::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let bitstream = (s as usize).view_bits_mut::<Lsb0>()[..segment_size].to_owned();
                
                // determining the new state is a bit more tricky than just looking
                // at the last bit of the segment:
                // 1. if the segment terminated with (11), this resets the state to 0
                //     xxxxxx11|1
                //     otherwise this would immediately terminate in [0]. But really the last bit
                //     has been used up in the terminator
                // 2. otherwise, we still have to be carefull:
                //     00000111|10000000
                //          OR
                //     000(11)(11)1|10000000
                //    Even though the last bits are 11, this is NOT a terminator
                //    and the next state indeed is 1
                //
                // The easiest way is to just see what comes back from the decoding:
                // We get the number of bits after the last terminator and can hence
                // pull out all the bits in questions

                                let r = decode_with_remainder(&bitstream, lastbit);
                let newstate;
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                if trailing_bits.len() == 0 {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    newstate = State(0);
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    newstate = if final_bit { State(1)} else {State(0)}
                }

                // insert result based on new state
                if lastbit {
                    table_state1.insert(s as u8,(newstate, r));
                }
                else{
                    table_state0.insert(s as u8,(newstate, r));
                }
            }
        }
        LookupU8Hash { table_state0, table_state1}
    }
}

impl U8Lookup for LookupU8Hash {
    fn lookup(&self, s: State, segment: u8) -> (State, &DecodingResult) {
        let (newstate, result) = match s {
            State(0) => self.table_state0.get(&(segment)).unwrap(),
            State(1) => self.table_state1.get(&(segment)).unwrap(),
            State(_) => panic!("yyy")
        };
        (*newstate, result)
    }
}

/// Fast Fibonacci decoding lookup table for 16bit segments
pub trait U16Lookup {
    /// given the state of the last decoding operation and the new segment, returns
    /// the (precomputed) new state and decoding result
    fn lookup(&self, s: State, segment: u16) -> (State, &DecodingResult);
}
/// Vector based lookup table for u8
pub struct LookupU16Vec {
    table_state0: Vec<(State, DecodingResult)>,
    table_state1: Vec<(State, DecodingResult)>,
}
impl LookupU16Vec {
    /// create a new Lookup table for fast fibonacci decoding using 16bit segments
    /// This implementation uses a vec
    pub fn new() -> Self {
        let segment_size = 16;

        let mut table_state0 = Vec::new();
        let mut table_state1 = Vec::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let bitstream = (s as usize).view_bits_mut::<Lsb0>()[..segment_size].to_owned();
                
                // determining the new state is a bit more tricky than just looking
                // at the last bit of the segment:
                // 1. if the segment terminated with (11), this resets the state to 0
                //     xxxxxx11|1
                //     otherwise this would immediately terminate in [0]. But really the last bit
                //     has been used up in the terminator
                // 2. otherwise, we still have to be carefull:
                //     00000111|10000000
                //          OR
                //     000(11)(11)1|10000000
                //    Even though the last bits are 11, this is NOT a terminator
                //    and the next state indeed is 1
                //
                // The easiest way is to just see what comes back from the decoding:
                // We get the number of bits after the last terminator and can hence
                // pull out all the bits in questions

                let r = decode_with_remainder(&bitstream, lastbit);
                let newstate;
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                if trailing_bits.len() == 0 {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    newstate = State(0);
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    newstate = if final_bit { State(1)} else {State(0)}
                }

                // insert result based on new state                 
                if lastbit {
                    table_state1.push((newstate, r))
                }
                else{
                    table_state0.push((newstate, r))
                }
            }
        }
        LookupU16Vec { table_state0, table_state1}
    }
}

impl U16Lookup for LookupU16Vec {
    fn lookup(&self, s: State, segment: u16) -> (State, &DecodingResult) {
        let (newstate, result) = match s {
            State(0) => self.table_state0.get(segment as usize).unwrap(),
            State(1) => self.table_state1.get(segment as usize).unwrap(),
            // State(0) => &self.table_state0[segment as usize],
            // State(1) => &self.table_state1[segment as usize],            
            State(_) => panic!("yyy")
        };
        (*newstate, result)
    }
}

/// hash based lookup table for u8
pub struct LookupU16Hash {
    table_state0: HashMap<u16, (State, DecodingResult)>,
    table_state1: HashMap<u16, (State, DecodingResult)>,
}
impl LookupU16Hash {
    /// create a new Lookup table for fast fibonacci decoding using 16bit segments
    /// This implementation uses a HashMap
    pub fn new() -> Self {
        let segment_size = 16;

        let mut table_state0 = HashMap::new();
        let mut table_state1 = HashMap::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let bitstream = (s as usize).view_bits_mut::<Lsb0>()[..segment_size].to_owned();
                
                // determining the new state is a bit more tricky than just looking
                // at the last bit of the segment:
                // 1. if the segment terminated with (11), this resets the state to 0
                //     xxxxxx11|1
                //     otherwise this would immediately terminate in [0]. But really the last bit
                //     has been used up in the terminator
                // 2. otherwise, we still have to be carefull:
                //     00000111|10000000
                //          OR
                //     000(11)(11)1|10000000
                //    Even though the last bits are 11, this is NOT a terminator
                //    and the next state indeed is 1
                //
                // The easiest way is to just see what comes back from the decoding:
                // We get the number of bits after the last terminator and can hence
                // pull out all the bits in questions

                let r = decode_with_remainder(&bitstream, lastbit);
                let newstate;
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                if trailing_bits.len() == 0 {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    newstate = State(0);
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    newstate = if final_bit { State(1)} else {State(0)}
                }

                // insert result based on new state
                if lastbit {
                    table_state1.insert(s as u16,(newstate, r));
                }
                else{
                    table_state0.insert(s as u16,(newstate, r));
                }
            }
        }
        LookupU16Hash { table_state0, table_state1}
    }
}

impl U16Lookup for LookupU16Hash {
    fn lookup(&self, s: State, segment: u16) -> (State, &DecodingResult) {
        let (newstate, result) = match s {
            State(0) => self.table_state0.get(&(segment)).unwrap(),
            State(1) => self.table_state1.get(&(segment)).unwrap(),
            State(_) => panic!("yyy")
        };
        (*newstate, result)
    }
}

#[cfg(test)]
mod testing_lookups {
    use bitvec::prelude::*;
    use super::*;
    use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn test_u8vec() {
        let t = LookupU8Vec::new();
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1].to_bitvec();
        let i = bits.load_be::<u8>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4], u:7, lu: 4})
        );

        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1].load_be::<u8>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2], u:7, lu: 4})
        );      
    }

    #[test]
    fn test_decode_u8_vec() {
        let t = LookupU8Vec::new();
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0].to_bitvec();

        assert_eq!(
            fast_decode_u8(bits, &t),
             vec![4, 109]
        );  
    }

    #[test]
    fn test_decode_u8_hash() {
        let t = LookupU8Hash::new();
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0].to_bitvec();

        assert_eq!(
            fast_decode_u8(bits, &t), 
             vec![4, 109]
        );  
    }

    #[test]
    fn test_decode_u16_vec() {
        let t = LookupU16Vec::new();
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0,   1,0,1, 1].to_bitvec();

        assert_eq!(
            fast_decode_u16(bits, &t), 
             vec![4, 109, 7]
        );  
    }

    #[test]
    fn test_decode_u16_hash() {
        let t = LookupU16Hash::new();
        let bits = bits![usize, Lsb0; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0,   1,0,1, 1].to_bitvec();

        assert_eq!(
            fast_decode_u16(bits, &t), 
             vec![4, 109, 7]
        );  
    }

    #[test]
    fn test_u8hash() {
        let t = LookupU8Hash::new();
        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1].load_be::<u8>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4], u:7, lu: 4})
        );

        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1].load_be::<u8>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2], u:7, lu: 4})
        );
    } 

    #[test]
    fn test_u16vec() {
        let t = LookupU16Vec::new();
        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4, 28], u:5, lu: 4})
        );

        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2, 28], u:5, lu: 4})
        );
    }
    #[test]
    fn test_u16hash() {
        let t = LookupU16Hash::new();
        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4, 28], u:5, lu: 4})
        );

        let i = bits![usize, Lsb0; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2, 28], u:5, lu: 4})
        );
    }    
}
