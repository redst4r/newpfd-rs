use bitvec::{vec::BitVec, prelude::Msb0, prelude::Lsb0};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
// use newpfd::fibonacci::{bitslice_to_fibonacci, bitslice_to_fibonacci2, bitslice_to_fibonacci3, bitslice_to_fibonacci4};
use newpfd::newpfd_bitvec::decode_fast;
use newpfd::newpfd_bitvec::{encode, decode};
use rand::distributions::{Distribution, Uniform};
use rand_distr::Geometric;

// type MyBitSlice = BitSlice<u8, Msb0>;
type MyBitVector = BitVec<u8, Msb0>;

/// Encoding/Decoding 1M random [0,255] numbers
fn newpfd_encode_decode(c: &mut Criterion){
    
    fn _dummy_encode(data: Vec<u64>) -> MyBitVector{
        let blocksize = 512;
        let enc = encode(data.into_iter(), blocksize);
        enc.0
    }

    fn _dummy_decode(data: MyBitVector, n_elements: usize) -> Vec<u64>{
        let blocksize = 512;
        let enc = decode(&data, n_elements, blocksize);
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
        |b| b.iter(|| _dummy_decode(black_box(enc.clone()), len))
    );

    // decoding eith fast fib
    fn _dummy_decode_fast(data: MyBitVector, n_elements: usize) -> Vec<u64>{

        let blocksize = 512;
        let enc = decode_fast(&data, n_elements, blocksize);
        enc.0
    }

    let len = data.len();
    let (enc, _) = encode(data.into_iter(), 512);
    c.bench_function(
        &format!("Fast NewPFD: Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode_fast(black_box(enc.clone()), len))
    );
}


criterion_group!(benches, newpfd_encode_decode);
criterion_main!(benches);