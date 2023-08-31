use bitvec::{vec::BitVec, prelude::Msb0};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fibonacci_codec::Encode;
use newpfd::fibonacci::{bits_from_table, FibonacciDecoder};
use newpfd::fibonacci_old::fib_enc_multiple;
use newpfd::{newpfd_bitvec::{encode, decode}, fibonacci::FIB64};
use rand::distributions::{Distribution, Uniform};


fn newpfd_encode_decode(c: &mut Criterion){
    
    fn _dummy_encode(data: Vec<u64>) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0>{

        let blocksize = 512;
        let enc = encode(data.into_iter(), blocksize);
        enc.0
    }

    fn _dummy_decode(data: BitVec<u8, Msb0>, n_elements: usize) -> Vec<u64>{

        let blocksize = 512;
        let enc = decode(&data, n_elements, blocksize);
        enc.0
    }

    let n = 1_000_000;
    let data_dist = Uniform::from(0..255);
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
    let (enc, _) = encode(data.into_iter(), 512);
    c.bench_function(
        &format!("Decoding {} elements", n),
        |b| b.iter(|| _dummy_decode(black_box(enc.clone()), len))
    );

}

#[allow(dead_code)]
fn fibonacci_encode(c: &mut Criterion){

    fn _dummy_my_fib(data: Vec<u64>) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0>{

        fib_enc_multiple(&data)
    }

    fn _dummy_bit_table(data: Vec<u64>) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0>{
        let mut overall: BitVec<u8, Msb0> = BitVec::new();

        for x in data {
            bits_from_table(x, FIB64, &mut overall).unwrap();
        }
        overall
    }   

    fn _dummy_fibonacci_codec(data: Vec<u64>) -> bit_vec::BitVec{
        data.fib_encode().unwrap()
    }   

    
    let n = 100_000;
    let data_dist = Uniform::from(1..255);
    let mut rng = rand::thread_rng();
    let mut data: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(data_dist.sample(&mut rng));
    } 

    c.bench_function(
        &format!("My Fibonacci Encoding {} elements", n),
        |b| b.iter(|| _dummy_my_fib(black_box(data.clone())))
    );

    c.bench_function(
        &format!("BitTable Fibonacci encoding {} elements", n),
        |b| b.iter(|| _dummy_bit_table(black_box(data.clone())))
    );
    c.bench_function(
        &format!("FibonacciCodec encoding {} elements", n),
        |b| b.iter(|| _dummy_fibonacci_codec(black_box(data.clone())))
    );
}

#[allow(dead_code)]
fn fibonacci_decode(c: &mut Criterion){

    fn _dummy_my_decode(data: BitVec<u8, Msb0>) -> Vec<u64>{
        let d = FibonacciDecoder::new(data.as_bitslice());
        let dec: Vec<_> = d.collect();
        dec
    }

    fn _dummy_fibonacci_codec_decode(data: bit_vec::BitVec) -> Vec<u64>{
        let x = fibonacci_codec::fib_decode_u64(data).map(|x|x.unwrap()).collect();
        x
    }

    let n = 1_000_000;
    let data_dist = Uniform::from(1..255);
    let mut rng = rand::thread_rng();
    let mut data: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(data_dist.sample(&mut rng));
    }
    
    let mut overall: BitVec<u8, Msb0> = BitVec::new();
    for &x in data.iter() {
        let mut enc =newpfd::fibonacci_old::fib_enc(x);
        overall.append(&mut enc);
    }
    c.bench_function(
        &format!("My Decoding {} elements", n),
        |b| b.iter(|| _dummy_my_decode(black_box(overall.clone())))
    );


    let enc  = data.fib_encode().unwrap();
    c.bench_function(
        &format!("Fib_Codec: Decoding {} elements", n),
        |b| b.iter(|| _dummy_fibonacci_codec_decode(black_box(enc.clone())))
    );

}

criterion_group!(benches, newpfd_encode_decode);
// criterion_group!(benches, fibonacci_encode, fibonacci_decode);
criterion_main!(benches);