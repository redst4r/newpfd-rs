//! r
use std::time::Instant;
use fastfibonacci::{bit_decode::MyBitVector, byte_decode::byte_manipulation::bits_to_fibonacci_generic_array_u32};
use newpfd::{decode, decode_fast_u16, decode_fast_u8, encode};
use rand::distributions::Distribution;
use rand_distr::Geometric;

///d
pub fn main(){
    let n = 10_000_000;

    let encoded_stream = random_stream(n);
    let now = Instant::now();
    let (dec, _)  = decode_fast_u8(&encoded_stream, n, 512);
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("decode_fast_u8\t\t\t {} in {:?}", dec.len(), elapsed_time);

    let now = Instant::now();
    let (dec, _)  = decode_fast_u16(&encoded_stream, n, 512);
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("decode_fast_u16\t\t\t {} in {:?}", dec.len(), elapsed_time);

    let now = Instant::now();
    let (dec, _)  = decode(&encoded_stream, n, 512);
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("decode\t\t\t {} in {:?}", dec.len(), elapsed_time);


    let bytes = bits_to_fibonacci_generic_array_u32(&encoded_stream);
    let now = Instant::now();
    let (dec, _) =newpfd::newpfd_u32::decode(&bytes, n, 512);
    let elapsed_time: std::time::Duration = now.elapsed();
    println!("newpfd_u32::decode\t\t\t {} in {:?}", dec.len(), elapsed_time);
}

fn random_stream(n: usize) -> MyBitVector {
    let data_dist = Geometric::new(0.01).unwrap();

    let mut rng = rand::thread_rng();
    let mut data: Vec<u64> = Vec::with_capacity(n);
    for _ in 0..n {
        data.push(data_dist.sample(&mut rng));
    }
    let (enc, _t) = encode(data.iter().cloned(), 512);
    enc
}


#[test]
fn testing_bitpacking() {
    use itertools::Itertools;
    use bitpacking::{BitPacker1x,BitPacker4x, BitPacker};

    let b = BitPacker4x::new();

    let raw_ints = (0..128).map(|x| x / 16).collect_vec();
    let num_bits = 3;

    let expected_bytes = num_bits * raw_ints.len() / 8;

    let mut compressed_buffer = vec![0; expected_bytes];

    let nbytes = b.compress(&raw_ints, compressed_buffer.as_mut_slice(), num_bits as u8);

    println!("n: {}, {:?}", nbytes, compressed_buffer);


    let expected_elements = nbytes * 8 / num_bits;
    let mut decompressed_buffer = vec![0_u32; expected_elements];
    let bytes_decoded = b.decompress(compressed_buffer.as_slice(), &mut decompressed_buffer, num_bits as u8);

    println!("n: {}, {:?}", bytes_decoded, decompressed_buffer);

    println!("{:b}", 100);
}