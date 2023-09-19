use bitvec::{vec::BitVec, prelude::Msb0, prelude::Lsb0};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fibonacci_codec::Encode;
use newpfd::fib_utils::random_fibonacci_stream;
use newpfd::fibonacci::{bits_from_table, FibonacciDecoder, bitslice_to_fibonacci, bitslice_to_fibonacci2, bitslice_to_fibonacci3, bitslice_to_fibonacci4};
use newpfd::fibonacci_fast::{fast_decode, fast_decode_u8, fast_decode_u16};
use newpfd::fibonacci_old::fib_enc_multiple;
use newpfd::{newpfd_bitvec::{encode, decode}, fibonacci::FIB64};
use rand::distributions::{Distribution, Uniform};

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

    fn _dummy_my_fib(data: Vec<u64>) -> MyBitVector{
        fib_enc_multiple(&data)
    }

    fn _dummy_bit_table(data: Vec<u64>) -> MyBitVector{
        let mut overall: MyBitVector = BitVec::new();

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

    fn _dummy_my_decode(data: MyBitVector) -> Vec<u64>{
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
    
    let mut overall: MyBitVector = BitVec::new();
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


fn fibonacci_bitslice(c: &mut Criterion){
    // let v: Vec<bool> = vec![1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1].iter().map(|x|*x==1).collect();
    let v: Vec<bool> = vec![1,0,1,0,1,0,1,1].iter().map(|x|*x==1).collect();
    let bs: MyBitVector  = BitVec::from_iter(v.into_iter());

    c.bench_function(
        &format!("fib_bitslice"),
        |b| b.iter(|| bitslice_to_fibonacci(black_box(&bs)))
    );

    c.bench_function(
        &format!("fib_bitslice2"),
        |b| b.iter(|| bitslice_to_fibonacci2(black_box(&bs)))
    );

    c.bench_function(
        &format!("fib_bitslice3"),
        |b| b.iter(|| bitslice_to_fibonacci3(black_box(&bs)))
    );
    c.bench_function(
        &format!("fib_bitslice4"),
        |b| b.iter(|| bitslice_to_fibonacci4(black_box(&bs)))
    );
}

fn fast_decode_vs_regular(c: &mut Criterion){

    // create a long fib string
    let data = random_fibonacci_stream(1_000_000, 1, 255);
    // make a copy for fast decoder
    let mut data_fast: BitVec<usize, Lsb0> = BitVec::new();
    for bit in data.iter().by_vals() {
        data_fast.push(bit);
    }

    c.bench_function(
        &format!("fast decode"),
        |b| b.iter(|| fast_decode(black_box(data_fast.clone()), black_box(8)))
    );

    c.bench_function(
        &format!("fast u8-decode"),
        |b| b.iter(|| fast_decode_u8(black_box(data_fast.clone())))
    );

    c.bench_function(
        &format!("fast u16-decode"),
        |b| b.iter(|| fast_decode_u16(black_box(data_fast.clone())))
    );


    fn dummy(bv: BitVec<u8, Msb0>) -> Vec<u64> {
        let dec = FibonacciDecoder::new(&bv);
        let x: Vec<_> = dec.collect();
        // println!("{}", x.len());
        x

    }
    c.bench_function(
        &format!("normal decode"),
        |b| b.iter(|| dummy(black_box(data.clone())))
    );

}

// criterion_group!(benches, newpfd_encode_decode);
// criterion_group!(benches, fibonacci_bitslice);

criterion_group!(benches, fast_decode_vs_regular);
// criterion_group!(benches, newpfd_encode_decode, fibonacci_encode, fibonacci_decode);
// criterion_group!(benches, fibonacci_decode);
criterion_main!(benches);