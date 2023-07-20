use bitvec::{vec::BitVec, prelude::Msb0};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use newpfd::newpfd_bitvec::{encode, decode};
use rand::distributions::{Distribution, Uniform};


fn plain_iterator_speed(c: &mut Criterion){
    
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


criterion_group!(benches, plain_iterator_speed);
criterion_main!(benches);