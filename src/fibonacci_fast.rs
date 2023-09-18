//!
//! 
use std::{collections::HashMap, hash::Hash};
use bitvec::{vec::BitVec, field::BitField, view::BitView};
use itertools::Itertools;
use num::traits::Pow;
use bitvec::prelude::*;
use rand::{distributions::Uniform, prelude::Distribution};
use crate::fibonacci::{MyBitSlice, FIB64, bits_from_table, MyBitVector, FibonacciDecoder};

struct Offset(usize);
struct LastBit(bool);


/// Problem:
/// if the entire segment doesnt contain a terminator, the value will not be added to the res
/// 

#[test]
fn create_lookup_old() {
    let table: HashMap<(Offset, LastBit, String), (Offset, LastBit, Vec<usize>)> = HashMap::new();

    let register_size = 8_usize;
    

    for i in 0..2.pow(register_size) {

        // turn i onto a vector of bools
        let segment = format!("{:08b}", i);
        let mut segment_bool = Vec::new();
        for s in segment.chars() {
            segment_bool.push(s=='1')
        }

        for offset in 0..1 {//register_size {
            for lastbit in vec![true, false]{
 
                let mut res = Vec::new();
                let mut o = offset;
                let mut l = lastbit;
                let mut accum = 0;

                for &b in segment_bool.iter() {
                    if b && l {
                        // found end
                        res.push(accum);  //store result
                        accum = 0;  //reset the accum for new number
                        //l = b;    // actually DONT set the last bit here!! otherwise 1111 -> [0,0,0]
                        l = false;
                        o = 0;  //reset the offset to a new starting nymber
                    }
                    else{
                        if b {
                            accum+= FIB64[o];
                        }
                        l = b;
                        o += 1;
                    }
                }

                res.push(accum);  // push whatever is left (some unfinished decoding in the right side)

                let new_offset = o;
                let new_lastbit = *segment_bool.last().unwrap();
                

                println!("Input: {lastbit}|{segment}  offset:{offset}");
                println!("O: {new_offset}  L {new_lastbit}  V {res:?}");
                println!("=================================")

            }
        }
    }
}



fn get_bits_for_state(State(s): State) -> (usize, usize) {
    let Lu = (s + 1 ).ilog2() ;  // ilog: less or equal than log2(x)
    let U = s - 2_usize.pow(Lu) + 1;
    (U, Lu as usize)
}

/// used to remember the position in the bitvector and the partically decoded number
#[derive(Debug, Clone, Eq, PartialEq, Hash, Copy)]
struct State(usize);

#[derive(Debug, Clone)]
struct FsmResult {
    numbers: Vec<u64>, // fully decoded numbers from the segment

    /// Stores the partially decoded number left in hte segment
    /// 
    U: u64,  // bits of partially decoded number; i.e. anything that didnt have a delimiter so far, kind of the accumulator
    Lu: usize,  //  bitlength of U; i..e how many bits processed so far without hitting the delimieter
}

use std::cmp;

fn create_lookup(segment_size: usize) -> HashMap<(State, BitVec<usize, Msb0>), (State, FsmResult)> {
    
    let mut table: HashMap<(State, BitVec<usize, Msb0>), (State, FsmResult)> = HashMap::new();

    for lastbit in vec![true, false]{
        for s in 0..2.pow(segment_size) {
            let bitstream = (s as usize).view_bits_mut::<Msb0>()[64-segment_size..].to_owned();
            
            let mut newstate = if *(bitstream.last().unwrap()) {State(1)} else {State(0)};
            // newstate is pretty easy (just the last bit)
            // EXCEPT this weird scenario:
            // SEGMENT1|SEGMENT2|
            // 00000011|10000000
            // i.e. a terminator flush with the sement border AND a following 1bit
            // in segment 2, if we set lastbit=True
            // we'd immediately emit a new number
            // however that lastbit has been "used up" in the delimiator already   
            // Hence: whenever we end with a terminator, set State=0
            let x1 = bitstream[bitstream.len()-2];
            let x2 = bitstream[bitstream.len()-1];
            if x1 && x2 {
                newstate = State(0);
            }
            
            // println!("{:?}", bitstream);
            let laststate = if lastbit {State(1)} else {State(0)};
            let r = decode_with_remainder(&bitstream, lastbit);
            table.insert((laststate, bitstream), (newstate, r));
        }
    }
    table
}

fn create_lookup_u8(segment_size: usize) -> HashMap<(State, u8), (State, FsmResult)> {
    
    let mut table: HashMap<(State, u8), (State, FsmResult)> = HashMap::new();

    for lastbit in vec![true, false]{
        for s in 0..2.pow(segment_size) {
            let bitstream = (s as usize).view_bits_mut::<Msb0>()[64-segment_size..].to_owned();
            
            let mut newstate = if *(bitstream.last().unwrap()) {State(1)} else {State(0)};
            // newstate is pretty easy (just the last bit)
            // EXCEPT this weird scenario:
            // SEGMENT1|SEGMENT2|
            // 00000011|10000000
            // i.e. a terminator flush with the sement border AND a following 1bit
            // in segment 2, if we set lastbit=True
            // we'd immediately emit a new number
            // however that lastbit has been "used up" in the delimiator already   
            // Hence: whenever we end with a terminator, set State=0
            let x1 = bitstream[bitstream.len()-2];
            let x2 = bitstream[bitstream.len()-1];
            if x1 && x2 {
                newstate = State(0);
            }
            
            // println!("{:?}", bitstream);
            let laststate = if lastbit {State(1)} else {State(0)};
            let r = decode_with_remainder(&bitstream, lastbit);
            table.insert((laststate, s as u8), (newstate, r));
        }
    }
    table
}

#[test]
fn test_create_lookup() {
    let t = create_lookup(8);
    println!("{:?}", t);
}

/// decodes a fibonacci stream until the very end of the stream
/// there might be a remainder (behind the last 11 delimiter)
/// which is also returned (its value in Fib, and its len in fib) 
fn decode_with_remainder(bitstream: &BitSlice<usize, Msb0>, lastbit_external: bool) -> FsmResult{

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
    return FsmResult {numbers: decoded_ints, U: accum, Lu: offset}
}

#[test]
fn test_decode_with_remainder_edge_case_delimiters() {

    // the exampel from the paper, Fig 9.4
    let bits = bits![usize, Msb0; 0,1,1,1,1];
    let r = decode_with_remainder(bits, false);
    assert_eq!(r.numbers, vec![2,1]);
    assert_eq!(r.U, 0);
    assert_eq!(r.Lu, 0);
}

#[test]
fn test_decode_with_remainder_edge_case_delimiters2() {

    // the exampel from the paper, Fig 9.4
    let bits = bits![usize, Msb0; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
    let r = decode_with_remainder(&bits, false);
    assert_eq!(r.numbers, vec![21,22]);
    assert_eq!(r.U, 0);
    assert_eq!(r.Lu, 0);
}

#[test]
fn test_decode_with_remainder_from_table94() {

    // the exampel from the paper, Fig 9.4
    let bits = &181.view_bits::<Msb0>()[64-8..];
    // let bits = bits![usize, Msb0; 0,1,1,1,1];
    let r = decode_with_remainder(bits, false);
    assert_eq!(r.numbers, vec![4]);
    assert_eq!(r.U, 7);
    assert_eq!(r.Lu, 4);

    // the exampel from the paper, Fig 9.4
    let bits = &165_usize.view_bits::<Msb0>()[64-8..];
    println!("{:?}", bits);
    let r = decode_with_remainder(&bits, true);
    assert_eq!(r.numbers, vec![0]);
    assert_eq!(r.U, 31);
    assert_eq!(r.Lu, 7);

    // the exampel from the paper, Fig 9.4
    let bits = &114_usize.view_bits::<Msb0>()[64-8..];
    let r = decode_with_remainder(&bits, true);
    assert_eq!(r.numbers, vec![2]);
    assert_eq!(r.U, 6);
    assert_eq!(r.Lu, 5);            
}


#[test]
fn test_decode_with_remainder() {

    // the exampel from the paper, Fig 9.4
    let bits = bits![usize, Msb0; 1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1];
    let r = decode_with_remainder(bits, false);
    assert_eq!(r.numbers, vec![4,7]);
    assert_eq!(r.U, 31);
    assert_eq!(r.Lu, 7);

    // the exampel from the paper, Fig 9.4
    // no remainder this time
    let bits = bits![usize, Msb0; 1,0,1,1,0,1,0,1,1];
    let r = decode_with_remainder(bits, false);
    assert_eq!(r.numbers, vec![4,7]);
    assert_eq!(r.U, 0);
    assert_eq!(r.Lu, 0);    
}

/// 
pub fn fast_decode(stream: BitVec<usize, Msb0>, segment_size: usize) -> Vec<u64> {

    // let the_table = create_lookup(segment_size);
    let the_table = create_lookup_u8(segment_size);

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

        // println!("===================================");
        // println!("=========== n={n} ===============");
        // println!("=========== len={len} ===============");
        // get a segment
        let start = i*segment_size;
        let end = cmp::min((i+1)*segment_size, stream.len());
        // let mut segment = stream[start..end].to_bitvec();
        // // the last segment might be shorter, we need to pad it
        // for i in 0..segment_size-segment.len() {
        //     segment.push(false);
        // }
        let segment = &stream[start..end];
        let segment_u8 = segment.load_be::<u8>();
        
        // println!("{state:?} {segment:?}", );
        // state machine
        // if !the_table.contains_key(&(state, segment)) {
        //     println!("{state:?}, {segment:?}");
        // }

        let (newstate, result) = the_table.get(&(state, segment_u8)).unwrap();
        // let (newstate, result) = the_table.get(&(state, segment)).unwrap();
        // let newstate = &State(0);
        // let result = FsmResult {numbers: vec![1], U:7, Lu:3};

        // println!("{newstate:?} {result:?}", );

        state = *newstate;
        let mutresult = result.clone();
        // 
        for num in mutresult.numbers {
            if len > 0 {  // some leftover from the previous segment
                // println!("{}", n);
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
        if result.Lu > 0 {
            n += fibonacci_left_shift(result.U, len);
            len += result.Lu;
        }
    }
    decoded_numbers
}


#[test]
fn test_fast_decode() {

    // the exampel from the paper, Fig 9.4
    let bits = bits![usize, Msb0; 1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,1,0,0,1,0].to_bitvec();
    let r = fast_decode(bits, 8);
    assert_eq!(r, vec![4,7, 86]);
}

#[test]
fn test_fast_decode_111_at_segment_border() {
    // edge case when the delimitator is algined with the segment and the next segment starts with 1
    // make sure to no double count 
    let bits = bits![usize, Msb0; 0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1].to_bitvec();
    let r = fast_decode(bits, 8);
    assert_eq!(r, vec![21, 22]);
}


#[test]
fn test_stuff() {
    let v: Vec<bool> = vec![1,0,1,1,0,1,1].iter().map(|x|*x==1).collect();
    let _b = BitVec::from_iter(v.into_iter());

    fast_decode(_b, 4);
}


fn fibonacci_left_shift_naive(number_in_binary: u64 ,k: usize) -> u64 {
    panic!("Buggy, doesnt yield the same as fibonacci_left_shift()");
    let bitrepresnetation = number_in_binary.view_bits::<Lsb0>();
    let mut accumulator = 0_u64;
    for (idx, current_bit) in bitrepresnetation.iter().by_vals().enumerate() {
        if current_bit {
            accumulator += FIB64[idx+k];
        }
    }
    accumulator
}


/// precompuated EXTENDED fibbonaci left shift by 1
/// n >> 1  = FIB_LEFT_1[n]
const FIB_EXT_LEFT_1: &[u64]= &[
0,1,1,2,3,3,4,4,5,6,
6,7,8,8,9,9,10,11,11,12,
12,13,14,14,15,16,16,17,17,18,
19,19,20,21,21,22,22,23,24,24,
25,25,26,27,27,28,29,29,30,30,
31,32,32,33,33
];

/// Theorem 3
/// Quick way to calculate the fibonacci left shift
fn fibonacci_left_shift(n: u64 ,k: usize) -> u64 {

    //  FIB64[k-1] * n + FIB64[k-2] + FIB_EXT_LEFT_1[n as usize];

    // we need F(-2), F(-1) which are not in the FIB64 array; here;s the workaround
    let (f_km1, f_km2) = match k {
        // Fib[-1], Fib[-2]
        0 => (1, 0),

        // Fib[0], Fib[-1]
        1 => (1, 1),

        // Fib[1], Fib[0]
        2 => (2, 1),

        // anything else
        k => (FIB64[k-1], FIB64[k-2])
    };

    let shifted = f_km1 * n + f_km2 * FIB_EXT_LEFT_1[n as usize];
    shifted
}


#[test]
fn test_fib_left() {
    // 32 = 0010101   (3+8+21)
    
    assert_eq!(
        fibonacci_left_shift(32, 0),
        32
    );

    assert_eq!(
        fibonacci_left_shift(32, 3),
        136
    );
}

fn dummy_encode(data: Vec<u64>) -> MyBitVector{
    let mut overall: MyBitVector = BitVec::new();

    for x in data {
        assert!(x>0);
        bits_from_table(x, FIB64, &mut overall).unwrap();
    } 
    overall
}
///
pub fn random_fibonacci_stream(n_elements: usize, min: usize, max: usize) -> MyBitVector{
    let data_dist = Uniform::from(min..max);
    let mut rng = rand::thread_rng();
    let mut data: Vec<u64> = Vec::with_capacity(n_elements);
    for _ in 0..n_elements {
        data.push(data_dist.sample(&mut rng) as u64);
    }
    dummy_encode(data)
}

#[test]
fn test_correctness() {
    let b = random_fibonacci_stream(200, 1, 100);
    // let b = dummy_encode(vec![64, 11, 88]);


    // make a copy for fast decoder
    let mut b_fast: BitVec<usize, Msb0> = BitVec::new();
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
fn test_fast_speed() {
    let b = random_fibonacci_stream(100000, 1, 10000);
    // make a copy for fast decoder
    let mut b_fast: BitVec<usize, Msb0> = BitVec::new();
    for bit in b.iter().by_vals() {
        b_fast.push(bit);
    }    
    let x2 = fast_decode(b_fast.clone(), 8);
    println!("{}", x2.iter().sum::<u64>())

}
/// just for debugging purpose
fn bitstream_to_string(buffer: &BitSlice<usize, Msb0>) -> String{
    let s = buffer.iter().map(|x| if *x{"1"} else {"0"}).join("");
    s
}