#![allow(missing_docs)]
use bitvec::{vec::BitVec, prelude::Msb0};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fastfibonacci::byte_decode::byte_manipulation::bits_to_fibonacci_generic_array_u32;
use newpfd::newpfd_bitvec::{decode_fast_u8, decode_fast_u16};
use newpfd::newpfd_bitvec::{encode, decode};
use rand::distributions::Distribution;
use rand_distr::Geometric;

type MyBitVector = BitVec<u8, Msb0>;

/// Encoding/Decoding 1M random [0,255] numbers
fn newpfd_encode_decode(c: &mut Criterion){
    
    fn _dummy_encode(data: Vec<u64>) -> MyBitVector{
        let blocksize = 512;
        let enc = encode(data.into_iter(), blocksize);
        enc.0
    }

    fn _dummy_decode(data: &MyBitVector, n_elements: usize) -> Vec<u64>{
        let blocksize = 512;
        let enc = decode(data, n_elements, blocksize);
        enc.0
    }

    let n = 1_000_000;
    let data_dist = Geometric::new(0.01).unwrap();

    let mut rng = rand::thread_rng();
    let mut data: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(data_dist.sample(&mut rng));
    }
    
    c.bench_function(
        &format!("Encoding {} elements", n),
        |b| b.iter(|| _dummy_encode(black_box(data.clone())))
    );

    // decoding
    let len = data.len();
    let (enc, _) = encode(data.iter().cloned(), 512);


    c.bench_function(
        &format!("Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode(black_box(&enc), len))
    );

    // decoding eith fast fib
    fn _dummy_decode_fast_u8(data: &MyBitVector, n_elements: usize) -> Vec<u64>{
        let blocksize = 512;
        let enc = decode_fast_u8(data, n_elements, blocksize);
        enc.0
    }

    c.bench_function(
        &format!("Fast-u8 NewPFD: Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode_fast_u8(black_box(&enc), len))
    );

    fn _dummy_decode_fast_u16(data: &MyBitVector, n_elements: usize) -> Vec<u64>{
        let blocksize = 512;
        let enc = decode_fast_u16(data, n_elements, blocksize);
        enc.0
    }

    c.bench_function(
        &format!("Fast-u16 NewPFD: Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode_fast_u16(black_box(&enc), len))
    );


    // decoding eith fast fib
    let bytes = bits_to_fibonacci_generic_array_u32(&enc);

    fn _dummy_decode_fast_u8_new(bytes: &[u8], n_elements: usize) -> Vec<u64>{
        let blocksize = 512;
        let enc = newpfd::newpfd_u32::decode(&bytes, n_elements, blocksize);
        enc.0
    }

    c.bench_function(
        &format!("New Fast-u8 NewPFD: Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode_fast_u8_new(black_box(&bytes), len))
    );
}


criterion_group!(benches, newpfd_encode_decode);
criterion_main!(benches);