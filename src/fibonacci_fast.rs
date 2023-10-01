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
//! and store the results in a vec, indexed by the segment. That is, we create a **Lookup table** (see [U8Lookup] and [U16Lookup] traits) which we can quickly query for incoming segments.
//! 
//! # Example
//! ```rust
//! use newpfd::fibonacci_fast::{fast_decode_u8,fast_decode_u16, LookupU8Vec, LookupU16Vec };
//! use bitvec::prelude as bv;
//! let bits = bv::bits![u8, bv::Msb0; 
//!     1,0,1,1,0,1,0,1,
//!     1,0,1,0,0,1,0,1,
//!     0,1,1,1,0,0,1,0].to_bitvec();
//! // using a u8 lookup table
//! let table = LookupU8Vec::new();
//! let r = fast_decode_u8(bits.clone(), &table);
//! assert_eq!(r, vec![4,7, 86]);
//! 
//! // using a u16 table
//! let table = LookupU16Vec::new();
//! let r = fast_decode_u16(bits.clone(), &table);
//! assert_eq!(r, vec![4,7, 86]);
//! ```

use std::{collections::{HashMap, VecDeque}, hash::Hash};
use bitvec::{vec::BitVec, field::BitField, view::BitView};
use itertools::Itertools;
use num::traits::Pow;
use bitvec::prelude::*;
use crate::{fibonacci::{FIB64, FbDec}, fib_utils::fibonacci_left_shift};
use std::cmp;

type FFBitorder = Msb0;

// the type of bitstream we expect as input!
type FFBitslice = BitSlice<u8, FFBitorder>;
use once_cell::sync::Lazy;

pub (crate) static FB_LOOKUP: Lazy<LookupU8Vec> = Lazy::new(|| {
    println!("initializing fibnoacci lookup");
    LookupU8Vec::new()
});

/// Fast Fibonacci decoding via an iterator. Let's you decode on element after the other,
/// whereas [fast_decode_u8], [fast_decode_u16] decode the entire stream at once. 
/// Using the iterator can be useful if the length of the binary-fibonacci code is not known, just the nubmer of elements.
/// 
/// # Example
/// ```rust,
/// use newpfd::fibonacci_fast::{FastFibonacciDecoder};
/// use bitvec::prelude as bv;
/// let bits = bv::bits![u8, bv::Msb0; 
///     1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,
///     0,1,1,1,0,0,1,0].to_bitvec();
/// // second argument allows for automatic shifing the decoded values by -1 
/// // (in case the encoding used +1 to be able to encode 0)
/// let f = FastFibonacciDecoder::new(&bits, false);
/// 
/// let x = f.collect::<Vec<_>>();
/// assert_eq!(x, vec![4,7, 86]);
/// 
/// // on can still use the remaining buffer for other things:
/// //use crate::newpfd::fibonacci::FbDec;
/// //let r = f.get_remaining_buffer();
/// //assert_eq!(r, &bits[19..]);        

/// ```
pub struct FastFibonacciDecoder<'a>  {
    bistream: &'a FFBitslice, // bitstream to decode
    position: usize,
    lookup_table: &'a LookupU8Vec,
    segment_size: usize,
    current_buffer: VecDeque<Option<u64>>,     // decoded numbers not yet emitted
    current_backtrack: VecDeque<Option<usize>>,  // for each decoded number in current_buffer, remember how many bits its encoding was
    state: State,
    decoding_state: DecodingState,
    // position in bitstream after the last emission; 
    // None if we exhausted the stream completely, 
    // i.e. the stream ended with a terminator and we emitted that item
    last_emission_last_position: Option<usize>, 
    shifted_by_one: bool,

}
impl <'a>  FastFibonacciDecoder<'a>  {
    /// New FastFibonacciDecoder
    /// # Parameters
    /// - bistream: (possibly infinite) sequence of bits containing fibonacci encoded u64s
    /// - shifted_by_one: if True, values are decremented by one (in case the encoding used a +1 shift to encode 0)
    pub fn new(bistream: &'a FFBitslice, shifted_by_one: bool) -> Self {
        FastFibonacciDecoder {
            bistream,
            position: 0,
            lookup_table: &FB_LOOKUP,
            current_buffer: VecDeque::new(),
            current_backtrack: VecDeque::new(),
            segment_size: u8::BITS as usize,
            state: State(0),
            decoding_state: DecodingState::new(),
            last_emission_last_position: None,
            shifted_by_one,
        }
    }

    /// decodes the next segment, adding to current_buffer and current_backtrack
    /// if theres nothgin to any more load (emtpy buffer, or partial buffer with some elements, but no terminator)
    /// it adds a None to `current_buffer` and `current_backtrack`, indicating the end of the iterator
    fn load_segment(&mut self) {
        // try to pull in the next segment
        // let segment = self.get_next_segment();
        let start = self.position;
        let end = cmp::min(self.position+self.segment_size, self.bistream.len());
        let segment = &self.bistream[start..end];
        // println!("Loading new segment: {}", bitstream_to_string(segment));
        self.position += self.segment_size; 

        if segment.is_empty() {
            // add the terminator into the iterator
            self.current_buffer.push_back(None);
            self.current_backtrack.push_back(None);
            return;
        }

        // decode this segment
        // NOTE: WE NEED TO PAD segments chunks that are smaller than segment_size ON THE RIGHT!!
        // this here pads depedingin on BitOrder: Lsb: right padding; Msb: left padding
        let mut segment_int = segment.load_be::<u8>();

        // to be generic over bitorder, we better do the padding ourselves!
        // turns out we can just do that with bitshifts on the integer
        // Msb:
        // pad   | 
        // 000000|1001101100
        // shift by 6
        // 1001101100|000000
        // Lsb
        // 1001101100|000000
        // no need to do anything!
        if segment.len() < self.segment_size {
            segment_int =  segment_int << (self.segment_size - segment.len());
        }        
    
        let (newstate, result) = self.lookup_table.lookup(self.state, segment_int);
        self.state = newstate;   
        self.decoding_state.update(result);

        // println!("Decoding state: {:?}", self.decoding_state);
        // move all decoded numbers into the emission buffer

        // todo: feels like this copy/move is unneeded, 
        // we could just pop of self.decoding_state.decoded_numbers directly when emitting for the iterator
        for el in self.decoding_state.decoded_numbers.drain(0..self.decoding_state.decoded_numbers.len()) {
            self.current_buffer.push_back(Some(el))
        }
        // move all decoded number-starts into the emission buffer
        // no need to delete in result, gets dropped
        for &el in result.number_end_in_segment.iter() {
            self.current_backtrack.push_back(Some(el+start)); // turn the relative (to segment start) into an absolute postion
        }            
        assert!(self.decoding_state.decoded_numbers.is_empty());

        // println!("current buffer: {:?}", self.current_buffer);

        if segment.len() != self.segment_size {
            // println!("partial segment");
            // a truncated segment, i.e. the last segment
            // push the terminator into the iterator to signal we;re done
            self.current_buffer.push_back(None);
            self.current_backtrack.push_back(None);
        }        
    }

}

impl <'a> FbDec<'a> for FastFibonacciDecoder<'a> {
    /// return the remaining bitstream after the last yielded number.
    /// If the bitstream is fully exhausted (i.e. no remainder), returns an empty bitslice.
    fn get_remaining_buffer(&self) -> &'a FFBitslice{
        match self.last_emission_last_position {
            Some(pos) =>  &self.bistream[pos+1..],
            None => self.bistream  // no single element was emitted, hence no bits of the stream affected
        }
    }

    /// Points to the first bit after the last terminator:
    /// ```bash
    /// ..011x...
    ///      ^
    /// ```
    /// If nothing was processed, or no single u64 could be decoded (no terminator), 
    /// points to the first bit in the string.
    fn get_bits_processed(&self) -> usize {
        match self.last_emission_last_position {
            Some(pos) =>  pos+1,
            None => 0, // no single element was emitted, hence no bits of the stream affected
        }
    }
}


/// The iterator is a bit complicated! We can't decode a single number 
/// (as the fast algorithm will potentially yield multiple numbers per segment), 
/// only an entire segment, hence we need to buffer the decoded numbers, yielding 
/// them one by one.
/// 
/// 
impl <'a>Iterator for FastFibonacciDecoder<'a> {
    type Item=u64;

    fn next(&mut self) -> Option<Self::Item> {

        while self.current_buffer.is_empty() { // could be that we need to load multiple segments until we gind a terminator!
            self.load_segment();
        }
        // } else {
        //     println!("non-empty buffer: just yielding what we have already: {:?} starts: {:?}", self.current_buffer, self.current_backtrack);
        // }
        // let el = self.current_buffer.remove(0); // TODO: expensive, Deque instead?
        // let _dummy = self.current_backtrack.remove(0); // TODO: expensive, Deque instead?
        // self.last_emission_last_position = _dummy;

        let el = self.current_buffer.pop_front().unwrap(); // unwrap should be save, there HAS to be an element
        let _dummy = self.current_backtrack.pop_front().unwrap();// unwrap should be save, there HAS to be an element
        
        // we only update the last elements position if we actually emit an item
        // if were just emitting the Iteration-TRM, the position of the last element remains unchanged
        if let Some(i) = _dummy {
            self.last_emission_last_position = Some(i);
        };

        // println!("State after emission: Position {}, LastPos {:?}, Buffer {:?} starts: {:?}", self.position, self.last_emission_last_position, self.current_buffer, self.current_backtrack);
        if self.shifted_by_one {
            el.map(|x| x-1)
        } else {
            el
        }
    }
}

#[cfg(test)]
mod test_iter {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_iter() {
        // just a 3number stream, trailing stuff
        let bits = bits![u8, FFBitorder; 
            1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,
            0,1,1,1,0,0,1,0].to_bitvec();
        let f = FastFibonacciDecoder::new(&bits, false);

        let x = f.collect::<Vec<_>>();
        assert_eq!(x, vec![4,7, 86]);
    }

    #[test]
    fn test_iter_multiple_segments() {

        // what if not a single number is decoded per segment
        let bits = bits![u8, FFBitorder; 
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0].to_bitvec();
        let mut f = FastFibonacciDecoder::new(&bits, false);

        // let x = f.collect::<Vec<_>>();
        assert_eq!(f.next(), Some(4181));
        assert_eq!(f.get_remaining_buffer(), &bits[19..]);


        let bits = bits![u8, FFBitorder; 
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0].to_bitvec();
        let mut f = FastFibonacciDecoder::new(&bits, false);

        // let x = f.collect::<Vec<_>>();
        assert_eq!(f.next(), Some(1597));
        assert_eq!(f.get_remaining_buffer(), &bits[17..]);        
    }

    #[test]
    fn test_iter_exact_borders() {

        // just a 3number stream, no trailing
        let bits = bits![u8, FFBitorder; 
            1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,  // 4, 7, 86
            0,1,1,1,0,0,1,1,0,1,1,1,0,0,1,1  // 3, 2, 4
        ].to_bitvec();

        let f = FastFibonacciDecoder::new(&bits, false);

        let x = f.collect::<Vec<_>>();
        assert_eq!(x, vec![4,7, 86, 6,2 ,6]);
    }
    #[test]
    fn test_iter_remaining_stream() {
        // just a 3number stream, no trailing
        let bits = bits![u8, FFBitorder; 
            1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,
            0,1,1,1,0,0,1,0].to_bitvec();
        
        let mut f = FastFibonacciDecoder::new(&bits, false);
        assert_eq!(f.next(), Some(4));
        assert_eq!(f.get_bits_processed(), 4);
        assert_eq!(f.get_remaining_buffer(), &bits[4..]);

        let mut f = FastFibonacciDecoder::new(&bits, false);
        assert_eq!(f.next(), Some(4));
        assert_eq!(f.next(), Some(7));
        assert_eq!(f.get_bits_processed(), 9);
        assert_eq!(f.get_remaining_buffer(), &bits[9..]);

        let mut f = FastFibonacciDecoder::new(&bits, false);
        assert_eq!(f.next(), Some(4));
        assert_eq!(f.next(), Some(7));
        assert_eq!(f.next(), Some(86));
        assert_eq!(f.get_bits_processed(), 19);
        assert_eq!(f.get_remaining_buffer(), &bits[19..]);

        let mut f = FastFibonacciDecoder::new(&bits, false);
        assert_eq!(f.next(), Some(4));
        assert_eq!(f.next(), Some(7));
        assert_eq!(f.next(), Some(86));
        assert_eq!(f.next(), None);
        assert_eq!(f.get_bits_processed(), 19);
        assert_eq!(f.get_remaining_buffer(), &bits[19..]);
    }

    #[test]
    fn test_iter_remaining_stream_nobits_processed() {
        // no terminator anywhere
        let bits = bits![u8, FFBitorder; 
        1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,
        0,1,0,1,0,0,1,0].to_bitvec();
        
        let mut f = FastFibonacciDecoder::new(&bits, false);
        assert_eq!(f.next(), None);
        assert_eq!(f.get_bits_processed(), 0);
        assert_eq!(f.get_remaining_buffer(), &bits);
    }
    #[test]
    fn test_correctness_iter() {
        use crate::fibonacci::FibonacciDecoder;
        use crate::fib_utils::random_fibonacci_stream;
        let b = random_fibonacci_stream(100000, 1, 1000);
        // let b = dummy_encode(vec![64, 11, 88]);
        // make a copy for fast decoder
        let mut b_fast: BitVec<u8, FFBitorder> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b, false);
        let x1: Vec<_> = dec.collect();
        
        let f = FastFibonacciDecoder::new(&b_fast, false);
        let x2: Vec<_> = f.collect();

        // println!("{:?}", bitstream_to_string(&b_fast));

        assert_eq!(x1, x2);
    }
}

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
    number_end_in_segment: Vec<usize>, // the position in the segment where that number started; undefined for the first number (as it couldve started in theh previous segment)
    /// Stores the partially decoded number left in hte segment
    u: u64,  // partially decoded number; i.e. anything that didnt have a delimiter so far, kind of the accumulator
    lu: usize,  // bitlength of U; i..e how many bits processed so far without hitting the delimieter
}

// lookup using the bitvec as an action on the statemachine
fn create_lookup(segment_size: usize) -> HashMap<(State, FFBitvec), (State, DecodingResult)> {
    
    let mut table: HashMap<(State, FFBitvec), (State, DecodingResult)> = HashMap::new();

    for lastbit in [true, false]{
        for s in 0..2.pow(segment_size) {
            // this one is dangerous, it depends on the bitorder!

            // MSB: we need the last few bits
            let t: usize = s.try_into().unwrap();
            let bb = t.view_bits::<FFBitorder>();
            let bitstream = bb[bb.len() - segment_size ..].to_owned();
            // bitstream.reverse();
            // LSB: we need the first few bits
            // let bitstream = (s as usize).view_bits::<FFBitorder>()[..segment_size].to_owned();
            
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
            
            // we need to know the bits behind the last terminator
            let trailing_bits= &bitstream[bitstream.len()-r.lu..];
            let newstate = if trailing_bits.is_empty() {
                // we ended with a temrinator:  xxxxxx11|yyy
                // resets the state to 0
                State(0)
            } else {
                let final_bit = trailing_bits[trailing_bits.len()-1];
                if final_bit { State(1)} else {State(0)}
            };

            let laststate = if lastbit {State(1)} else {State(0)};
            
            // converting to u8 storage
            let mut b:BitVec<u8, FFBitorder> = BitVec::new();
            b.extend(bitstream);
            // println!("Inserting {}", b);
            table.insert((laststate, b), (newstate, r));
        }
    }
    table
}

/// decodes a fibonacci stream until the very end of the stream
/// there might be a remainder (behind the last 11 delimiter)
/// which is also returned (its value in Fib, and its len in fib) 
fn decode_with_remainder<T: BitStore,O:BitOrder>(bitstream: &BitSlice<T,O>, lastbit_external: bool) -> DecodingResult{

    assert!(bitstream.len() < 64, "fib-codes cant be longer than 64bit, something is wrong!");
    // println!("{}", bitstream_to_string(bitstream));
    let mut prev_bit = lastbit_external;
    let mut decoded_ints = Vec::new();
    let mut decoded_end_pos: Vec<usize> = Vec::new();

    // let mut current_start = None; // already add the unknown starto f the first el

    let mut accum = 0_u64;
    let mut offset = 0;
    for (ix, b) in bitstream.iter().by_vals().enumerate() {
        match (prev_bit, b) {
            (false, true) => {
                accum+= FIB64[offset];
                offset+=1;
                prev_bit = b;
            }
            (true, true) => {
                // found delimiter; Note, the bit has already been counted in acc
                decoded_ints.push(accum);
                decoded_end_pos.push(ix); // will be none the first iteration
                // reset for next number
                accum = 0;
                offset = 0;
                // BIG TRICK: we need to set lastbit=false for the next iteration; otherwise 011|11 picks up the next 11 as delimiter
                prev_bit=false;
            }
            (false, false) | (true, false) => {
                // nothing to add, jsut increase offset
                offset+=1;
                prev_bit = b;
            }
        }
    }
    DecodingResult {numbers: decoded_ints, u: accum, lu: offset, number_end_in_segment: decoded_end_pos}
}

#[cfg(test)]
mod test_decode_with_remainder{
    use super::*;
    use pretty_assertions::assert_eq;
    #[test]
    fn test_decode_with_remainder_edge_case_delimiters() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![u8, FFBitorder; 0,1,1,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(
            r, 
            DecodingResult {numbers: vec![2,1], u: 0, lu:0, number_end_in_segment: vec![2,4]} 
        );
    }

    #[test]
    fn test_decode_with_remainder_edge_case_delimiters2() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![u8, FFBitorder; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
        let r = decode_with_remainder(&bits, false);
        assert_eq!(
            r, 
            DecodingResult {numbers: vec![21,22], u: 0, lu:0, number_end_in_segment: vec![7,15]} 
        );        
    }

    // #[test]
    fn test_decode_with_remainder_from_table94() {

        // the exampel from the paper, Fig 9.4
        let mut bits = &181_u8.view_bits::<FFBitorder>()[..8];
        // bits.reverse();
        
        println!("{bits:?}");
        // let bits = bits![u8, FFBitorder; 0,1,1,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(r.numbers, vec![4]);
        assert_eq!(r.u, 7);
        assert_eq!(r.lu, 4);

        // the exampel from the paper, Fig 9.4
        let bits = &165_u8.view_bits::<FFBitorder>()[..8];
        println!("{:?}", bits);
        let r = decode_with_remainder(&bits, true);
        assert_eq!(r.numbers, vec![0]);
        assert_eq!(r.u, 31);
        assert_eq!(r.lu, 7);

        // the exampel from the paper, Fig 9.4
        let bits = &114_u8.view_bits::<FFBitorder>()[..8];
        let r = decode_with_remainder(&bits, true);
        assert_eq!(r.numbers, vec![2]);
        assert_eq!(r.u, 6);
        assert_eq!(r.lu, 5);            
    }


    #[test]
    fn test_decode_with_remainder() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(
            r, 
            DecodingResult {numbers: vec![4,7], u: 31, lu:7, number_end_in_segment: vec![3, 8,]} 
        );   

        // the exampel from the paper, Fig 9.4
        // no remainder this time
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,1];
        let r = decode_with_remainder(bits, false);
        assert_eq!(
            r, 
            DecodingResult {numbers: vec![4,7], u: 0, lu:0, number_end_in_segment: vec![3, 8]} 
        );  
    }
    #[test]
    fn test_decode_with_remainder_starts() {
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,1, 0,1,1,0,0,0];
        let r = decode_with_remainder(bits, false);
        assert_eq!(
            r, 
            DecodingResult {numbers: vec![4,7,2], u: 0, lu:3, number_end_in_segment: vec![3,8,11]} 
        );  
    }
}

/// State of the decoding across multiple segments. Keeps track of completely 
/// decoded and partially decoded numbers (which could be completed with the next segment)
/// 
#[derive(Debug)]
struct DecodingState {
    decoded_numbers: Vec<u64>,  // completly decoded numbers across all segments so far
    n: u64,   // the partially decoded number from the last segment
    len: usize, // length of the last partially decoded number in bits; needed to bitshift whatever the next decoding result is
}
impl DecodingState {
    pub fn new() -> Self {
        DecodingState {decoded_numbers: Vec::new(), n: 0, len: 0 }
    }

    /// update the current decoding state with the results from the last segment of bits;
    /// i.e. new decoded numbers, new accumulator (n) and new trailing bits (len) 
    fn update(&mut self, decoding_result: &DecodingResult) {
        for &num in decoding_result.numbers.iter() {
            if self.len > 0 {  // some leftover from the previous segment
                self.n += fibonacci_left_shift(num, self.len);
            } else {
                self.n = num;
            }

            self.decoded_numbers.push(self.n);
            self.n = 0;
            self.len = 0;
        }

        // the beginning and inner port of F(n)
        if decoding_result.lu > 0 {
            self.n += fibonacci_left_shift(decoding_result.u, self.len);
            self.len += decoding_result.lu;
        }
    }
}

/// Reference implementation, very inefficient though. Use [fast_decode_u8] or [fast_decode_u16]
/// for actual applications.
pub fn fast_decode(stream: FFBitvec, segment_size: usize) -> Vec<u64> {

    let the_table = create_lookup(segment_size);
    let mut decoding_state = DecodingState::new();

    // let mut decoded_numbers = Vec::new();
    let mut n_chunks = stream.len() / segment_size;
    if stream.len() % segment_size > 0 {
        n_chunks+= 1;
    } 

    let mut state = State(0);

    // for segment in &stream.iter().chunks(8) {
    for i in 0..n_chunks{
        // get a segment
        let start = i*segment_size;
        let end = cmp::min((i+1)*segment_size, stream.len());
        let mut segment = stream[start..end].to_bitvec();

        // the last segment might be shorter, we need to pad it
        for _ in 0..segment_size-segment.len() {
            segment.push(false);
        }
        
        // println!("{state:?} {segment:?}", );
        // state machine
        if !the_table.contains_key(&(state, segment.clone())) {
            println!("{state:?}, {segment:?}");
        }

        // let debug = segment.clone();
        let (newstate, result) = the_table.get(&(state, segment)).unwrap();
            // .unwrap_or_else(|| panic!("Hashmap doesnt have {:?} {:?}, Hashmap length {}", state, debug, the_table.len()));

        state = *newstate;
        // update the current decoding state:
        // add completely decoded numbers, deal with partially decoded numbers
        decoding_state.update(result);
    }
    decoding_state.decoded_numbers
}

/// Decode a fibonacci stream using the Fast algorithm operating on 8bit segments.
// We explicityl chop the bitstream into u8-segments
// -> each segment can be represented by a number, which solves the problem of slow hashmap access
pub fn fast_decode_u8(stream: FFBitvec, table: &impl U8Lookup) -> Vec<u64> {

    let segment_size = 8;
    let mut decoding_state = DecodingState::new();
    // let mut n_chunks = stream.len() / segment_size;
    // if stream.len() % segment_size > 0 {
    //     n_chunks+= 1;
    // } 

    let mut state = State(0);
    // for i in 0..n_chunks{
    for segment in stream.chunks(segment_size) {
        // // get a segment
        // let start = i*segment_size;
        // let end = cmp::min((i+1)*segment_size, stream.len());
        // let segment = &stream[start..end];

        // NOTE: WE NEED TO PAD segments chunks that are smaller than segment_size ON THE RIGHT!!
        // this here pads depedingin on BitOrder: Lsb: right padding; Msb: left padding

        let mut segment_u8 = segment.load_be::<u8>();

        // to be generic over bitorder, we better do the padding ourselves!
        // turns out we can just do that with bitshifts on the integer
        // Msb:
        // pad   | 
        // 000000|1001101100
        // shift by 6
        // 1001101100|000000
        // Lsb
        // 1001101100|000000
        // now need to do anything!
        if segment.len() < segment_size {
            segment_u8 =  segment_u8 << (segment_size - segment.len());
        }

        let (newstate, result) = table.lookup(state, segment_u8);
        state = newstate;
        // update the current decoding state:
        // add completely decoded numbers, deal with partially decoded numbers
        decoding_state.update(result);
    }
    decoding_state.decoded_numbers
}

/// Decode a fibonacci stream using the Fast algorithm operating on 16bit segments.
// We explicityl chop the bitstream into u16-segments
// -> each segment can be represented by a number, which solves the problem of slow hashmap access
pub fn fast_decode_u16(stream: FFBitvec, table: &impl U16Lookup) -> Vec<u64> {

    // println!("Total stream {}", bitstream_to_string(&stream));
    let segment_size = 16;
    let mut decoding_state = DecodingState::new();

    // let mut n_chunks = stream.len() / segment_size;
    // if stream.len() % segment_size > 0 {
    //     n_chunks+= 1;
    // } 

    let mut state = State(0);
    // for i in 0..n_chunks{
    for segment in stream.chunks(segment_size) {
        // get a segment
        // let start = i*segment_size;
        // let end = cmp::min((i+1)*segment_size, stream.len());
        // let segment = &stream[start..end];

        // NOTE: WE NEED TO PAD segments chunks that are smaller than segment_size ON THE RIGHT!!
        // this here pads depedingin on BitOrder: Lsb: right padding; Msb: left padding
        let mut segment_u16 = segment.load_be::<u16>();

        // to be generic over bitorder, we better do the padding ourselves!
        // turns out we can just do that with bitshifts on the integer
        // Msb:
        // pad   | 
        // 000000|1001101100
        // shift by 6
        // 1001101100|000000
        // Lsb
        // 1001101100|000000
        // now need to do anything!
        if segment.len() < segment_size {
            segment_u16 =  segment_u16 << (segment_size - segment.len());
        }

        let (newstate, result) = table.lookup(state, segment_u16);

        // println!("Segment: {}, SegmentInt {}", segment, segment_u16);
        // println!("Decoded state: {:?}, result {:?}", newstate, result);

        state = newstate;
        // update the current decoding state:
        // add completely decoded numbers, deal with partially decoded numbers        
        decoding_state.update(result);
        // println!("Updated state: {:?}", decoding_state);

    }
    decoding_state.decoded_numbers
}

#[cfg(test)]
mod testing_fast_decode {
    use super::*;
    use pretty_assertions::assert_eq;
    
    #[test]
    fn test_fast_decode_correct_padding() {
        // if the chunk is smaller than the segmentsize, it becomes important how we pad it back to 
        // sgementsize
        // the exampel from the paper, Fig 9.4
        let bits = bits![u8, FFBitorder; 
            1,0,1,1].to_bitvec();

        let r = fast_decode(bits.clone(), 8);
        assert_eq!(r, vec![4]);

        let table = LookupU8Vec::new();
        let r = fast_decode_u8(bits.clone(), &table);
        assert_eq!(r, vec![4]);

        let table = LookupU16Vec::new();
        let r = fast_decode_u16(bits, &table);
        assert_eq!(r, vec![4]);    
    }

    #[test]
    fn test_fast_decode() {

        // the exampel from the paper, Fig 9.4
        let bits = bits![u8, FFBitorder; 
            1,0,1,1,0,1,0,1,
            1,0,1,0,0,1,0,1,
            0,1,1,1,0,0,1,0].to_bitvec();

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
        let bits = bits![u8, FFBitorder; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
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

        let bits = bits![u8, FFBitorder; 
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
        let bits = bits![u8, FFBitorder; 
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
        let b = random_fibonacci_stream(100000, 1, 1000);
        // let b = dummy_encode(vec![64, 11, 88]);
        // make a copy for fast decoder
        let mut b_fast: BitVec<u8, FFBitorder> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b, false);
        let x1: Vec<_> = dec.collect();
        let x2 = fast_decode(b_fast.clone(), 8);

        // println!("{:?}", bitstream_to_string(&b_fast));

        assert_eq!(x1, x2);
    }

    #[test]
    fn test_correctness_fast_decode_u8() {
        use crate::fibonacci::FibonacciDecoder;
        use crate::fib_utils::random_fibonacci_stream;
        let b = random_fibonacci_stream(100000, 1, 1000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<u8, FFBitorder> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b, false);
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
        let b = random_fibonacci_stream(1000000, 1, 1000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<u8, FFBitorder> = BitVec::new();
        for bit in b.iter().by_vals() {
            b_fast.push(bit);
        }
        let dec = FibonacciDecoder::new(&b, false);
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

        let b = random_fibonacci_stream(10_000_000, 100000, 1000000);
        // make a copy for fast decoder
        let mut b_fast: BitVec<u8, FFBitorder> = BitVec::new();
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
        println!("{}", x2.iter().sum::<u64>())    ;


        // let x2 = fast_decode(b_fast.clone(), 8);
        let f = FastFibonacciDecoder::new(&b_fast, false);
        println!("{}", f.sum::<u64>())

    }
}

/// just for debugging purpose
pub fn bitstream_to_string<T:BitStore, O:BitOrder>(buffer: &BitSlice<T, O>) -> String{
    let s = buffer.iter().map(|x| if *x{"1"} else {"0"}).join("");
    s
}

/// Fast Fibonacci decoding lookup table for 8bit segments
pub trait U8Lookup {
    /// Given the state of the last decoding operation and the new segment, returns
    /// the (precomputed) new state and decoding result.
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
        let segment_size = 8_usize;

        let mut table_state0 = Vec::new();
        let mut table_state1 = Vec::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let t: u8 = s.try_into().unwrap();

                // dangerous, depends on bitorder!!
                // Msb: need to reveverse the bits, but if we keep it consistent with loading/storing, should be fine
                let bitstream = t.view_bits::<FFBitorder>().to_owned();
                assert_eq!(bitstream.len(), 8);

                //Lsb: no need to change anything
                // let bitstream = t.view_bits::<FFBitorder>()[..segment_size].to_owned();
                
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
                
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                let newstate = if trailing_bits.is_empty() {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    State(0)
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    if final_bit { State(1)} else {State(0)}
                };

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
        let segment_size = 8_usize;

        let mut table_state0 = HashMap::new();
        let mut table_state1 = HashMap::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let t: u8 = s.try_into().unwrap();
                let bitstream = t.view_bits::<FFBitorder>().to_owned();
                assert_eq!(bitstream.len(), 8);

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
                
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                let newstate = if trailing_bits.is_empty() {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    State(0)
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    if final_bit { State(1)} else {State(0)}
                };

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
#[derive(Debug)]
pub struct LookupU16Vec {
    table_state0: Vec<(State, DecodingResult)>,
    table_state1: Vec<(State, DecodingResult)>,
}
impl LookupU16Vec {
    /// create a new Lookup table for fast fibonacci decoding using 16bit segments
    /// This implementation uses a vec
    pub fn new() -> Self {
        let segment_size = 16_usize;

        let mut table_state0 = Vec::new();
        let mut table_state1 = Vec::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let t: u16 = s.try_into().unwrap();
                let bitstream = t.view_bits::<FFBitorder>().to_owned();
                assert_eq!(bitstream.len(), 16);
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
                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                let newstate = if trailing_bits.is_empty() {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    State(0)
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    if final_bit { State(1)} else {State(0)}
                };

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
        let segment_size = 16_usize;

        let mut table_state0 = HashMap::new();
        let mut table_state1 = HashMap::new();
    
        for lastbit in [true, false]{
            for s in 0..2.pow(segment_size) {
                let t: u16 = s.try_into().unwrap();
                let bitstream = t.view_bits::<FFBitorder>().to_owned();
                assert_eq!(bitstream.len(), 16);
                
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

                // we need to know the bits behind the last terminator
                let trailing_bits= &bitstream[bitstream.len()-r.lu..];
                let newstate = if trailing_bits.is_empty() {
                    // we ended with a temrinator:  xxxxxx11|yyy
                    // resets the state to 0
                    State(0)
                } else {
                    let final_bit = trailing_bits[trailing_bits.len()-1];
                    if final_bit { State(1)} else {State(0)}
                };

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
    // use pretty_assertions::{assert_eq, assert_ne};

    #[test]
    fn test_u8vec() {
        let t = LookupU8Vec::new();
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1].to_bitvec();
        let i = bits.load_be::<u8>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4], u:7, lu: 4, number_end_in_segment: vec![3]})
        );

        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1].load_be::<u8>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2], u:7, lu: 4, number_end_in_segment: vec![0, 3]})
        );      
    }

    #[test]
    fn test_u8hash() {
        let t = LookupU8Hash::new();
        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1].load_be::<u8>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4], u:7, lu: 4, number_end_in_segment: vec![3,]})
        );

        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1].load_be::<u8>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2], u:7, lu: 4, number_end_in_segment: vec![0, 3]})
        );
    } 

    #[test]
    fn test_u16vec() {
        let t = LookupU16Vec::new();
        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4, 28], u:5, lu: 4, number_end_in_segment: vec![3, 11]})
        );

        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2, 28], u:5, lu: 4, number_end_in_segment: vec![0,3,11]})
        );
    }
    #[test]
    fn test_u16hash() {
        let t = LookupU16Hash::new();
        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();

        assert_eq!(
            t.lookup(State(0), i), 
            (State(1), &DecodingResult {numbers: vec![4, 28], u:5, lu: 4, number_end_in_segment: vec![3,11]})
        );

        let i = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1, 0,0,1,1,0,0,0,1].load_be::<u16>();
        assert_eq!(
            t.lookup(State(1), i), 
            (State(1), &DecodingResult {numbers: vec![0, 2, 28], u:5, lu: 4, number_end_in_segment: vec![0,3,11]})
        );
    }
    #[test]
    fn test_decode_u8_vec() {
        let t = LookupU8Vec::new();
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0].to_bitvec();

        assert_eq!(
            fast_decode_u8(bits, &t),
             vec![4, 109]
        );  
    }

    #[test]
    fn test_decode_u8_hash() {
        let t = LookupU8Hash::new();
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0].to_bitvec();

        assert_eq!(
            fast_decode_u8(bits, &t), 
             vec![4, 109]
        );  
    }

    #[test]
    fn test_decode_u16_vec() {
        let t = LookupU16Vec::new();
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0,   1,0,1, 1].to_bitvec();

        assert_eq!(
            fast_decode_u16(bits, &t), 
             vec![4, 109, 7]
        );  
    }

    #[test]
    fn test_decode_u16_hash() {
        let t = LookupU16Hash::new();
        let bits = bits![u8, FFBitorder; 1,0,1,1,0,1,0,1,   0, 1, 0, 0, 0, 1, 1, 0,   1,0,1, 1].to_bitvec();

        assert_eq!(
            fast_decode_u16(bits, &t), 
             vec![4, 109, 7]
        );  
    }

    
}
