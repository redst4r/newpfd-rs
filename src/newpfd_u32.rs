//! Same as [`crate::newpfd_bitvec`], but instead of using the `bitvec` backend
//! lets work directly with a bytestream (coming from a stream of u32, as in bustools)
//! 
//! Note: this implies a specific way the bytes have been stored! Usually (e.g. bustools) those bytes
//! come in chunks of 4 (i.e. from a `u32`) and are stored in LittleEndian (implying that we need to look at 
//! `newpfd_buf[3]`, then `newpfd_buf[2]`, ...).
//! This transformation of the bytestream happens via [FastFibonacciDecoderNewU8].

use bitvec::prelude as bv;
use fastfibonacci::byte_decode::{bytestream_transform::U32BytesToU8, faster::{FastFibonacciDecoderNewU8, FB_LOOKUP_NEW_U8}};
use itertools::{izip, Itertools};
use crate::{newpfd_bitvec::PrimaryBuffer, MyBitStore};

///  decode a buffer cointaining `n_elements` integeres which are NewPDF-compressed with specified `blocksize`.
/// - `blocksize` max number of elements in the block. There CAN be less.
pub fn decode(newpfd_buf: &[u8], n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){
    decode_general(newpfd_buf, n_elements, blocksize)
}

// /// Decode a NewPFD-encoded buffer, containing `n_elements`. Alias for [decode], see its documentation
// pub fn decode_normal(newpfd_buf: &[u8], n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){
//     decode_general(newpfd_buf, n_elements, blocksize, FibDecodeMode::Normal)
// }

// /// Decode a NewPFD-encoded buffer, containing `n_elements`. Uses a `FastFibonacci<u8>` Decoder in the background. See [`decode`].
// pub fn decode_fast_u8(newpfd_buf: &[u8], n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){
//     decode_general(newpfd_buf, n_elements, blocksize, FibDecodeMode::FastU8)
// }

// /// Decode a NewPFD-encoded buffer, containing `n_elements`. Uses a `FastFibonacci<u16>` Decoder in the background. See [`decode`].
// pub fn decode_fast_u16(newpfd_buf: &[u8], n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){
//     decode_general(newpfd_buf, n_elements, blocksize, FibDecodeMode::FastU16)
// }

/// decode a NewPFD block, either with a regular Fibonacci Decoder (`mode=FibDecodeMode::Normal`) 
/// or the Fast Fibonacci Decode (`FibDecodeMode::FastU8`, `FibDecodeMode::FastU16`)
/// TODO: performance: `elements` as a reusable buffer that gets modified?
fn decode_general(newpfd_buf: &[u8], n_elements: usize, blocksize: usize, /* mode: FibDecodeMode */) -> (Vec<u64>, usize){

    let mut pos = 0;
    let mut elements: Vec<u64> = Vec::with_capacity(n_elements);

    while elements.len() < n_elements {
        // each call shortens the encoded BitVec
        let current_block = &newpfd_buf[pos..];
        let (mut els, bytes_consumed) = decode_newpfdblock(current_block, blocksize);
        
        pos += bytes_consumed;
        elements.append(&mut els);
    }
    // trucate, as we retrieved a bunch of zeros from the last block
    elements.truncate(n_elements);

    (elements, pos)
}

fn decode_newpfdblock(buf: &[u8], blocksize: usize, /*mode:FibDecodeMode*/) -> (Vec<u64>, usize) {

    let mut buf_position = 0;

    // let mut fibdec:  Box<dyn FbDec> = match mode{
    //     FibDecodeMode::FastU8 =>   Box::new(fastfibonacci::bit_decode::fast::get_u8_decoder(buf, true)),
    //     FibDecodeMode::FastU16 =>   Box::new(fastfibonacci::bit_decode::fast::get_u16_decoder(buf, true)),
    //     FibDecodeMode::Normal => Box::new(fibonacci::FibonacciDecoder::new(buf,true)),
    // };
    let mut fibdec: FastFibonacciDecoderNewU8<&[u8]> = FastFibonacciDecoderNewU8::new(
        buf,
        &FB_LOOKUP_NEW_U8,
        true,
        fastfibonacci::byte_decode::faster::StreamType::U32
    );


    // let mut fibdec = FastFibonacciDecoder::new(buf, true);
    // let mut fibdec = fibonacci::FibonacciDecoder::new(buf,true);

    // pulling the elements out of the header (b_bits, min_el, n_exceptions)
    let (_b_bits, min_el, n_exceptions) = fibdec.by_ref().take(3).next_tuple().unwrap(); //TODO yield a protocolError if we cant get 3 elements
    let b_bits = _b_bits as usize;
    
    // println!("bits {b_bits} min_el {min_el}, n_exp {n_exceptions}");

    // decoding the Gaps
    let index_gaps: Vec<u64> = fibdec.by_ref().take(n_exceptions as usize).collect();
    assert_eq!(index_gaps.len(), n_exceptions as usize, "protocol error, not enough exceptions");

    // Decoding expections
    let exceptions: Vec<u64> = fibdec.by_ref().take(n_exceptions as usize).map(|x|x+1).collect();
    assert_eq!(exceptions.len(), n_exceptions as usize, "protocol error, not enough exceptions");

    // turn index gaps into the actual index (of expections)
    let index: Vec<u64> = index_gaps
        .into_iter()
        .scan(0, |acc, i| {
            *acc += i;
            Some(*acc)
        })
        .collect();

    // println!("index: {index:?}");
    // println!("exceptions: {exceptions:?}");


    // need to remove trailing 0s which where used to pad to a muitple of u32
    let nbytes = fibdec.get_consumed_bytes();
    let delta_bytes = nbytes.next_multiple_of(4); // TODO not needed   
    let padded_bytes =  delta_bytes - nbytes;

    // those bytes should all be zero
    let zeros = &buf[buf_position+nbytes..buf_position + nbytes + padded_bytes];
    assert!(zeros.iter().all(|x| *x==0)) ;
    
    // move the new buffer position at the end
    buf_position += delta_bytes;

    // the body of the block, i.e. bitpacked integers
    let buf_body = &buf[buf_position..];

    // println!("Pody pos {buf_position}");
    // println!("body: {:?}", buf_body);

    // note that a block can be shorter than sepcifiec if we ran out of elements
    // however, as currently implements (by bustools)
    // the primary block always has blocksize*b_bits bitsize!
    // i.e. for a partial block, we cant really know how many elements are in there
    // (they might all be 0)
    // hence lets get the predefined size of the primary blokc:

    // we need to hget the primary_buffer into Msb Order (required!!)
    // NOTE THAT CONVERSION IS VERY SLOW!!! I.e. MSB->LSB copy takes much longer than a MSB->MSB copy
    // i.e. dont do the following; but rather have everything work with Msb
    //
    // let mut buf_body_full: bv::BitVec<u8, bv::Msb0> = bv::BitVec::with_capacity(blocksize*b_bits);
    // buf_body_full.extend_from_bitslice(&buf[buf_position..buf_position+(blocksize*b_bits)]);

    // as opposed to workng with the bitstream directly (see newpdf_bitvec),
    // when working with the bytes, we need to get the groups of 4bytes into the right
    // order
    //
    // TODO: we could probably also get get the correct number of bytes (blocksize * bits)
    // convert to u32 and feed it into `bv::BitSlice::from_slice`
    let correct_bytestream = U32BytesToU8::new(buf_body);
    // luckily, we know how many bits are in the body: blocksize * b_bits
    // rounded up to the next u32
    let n_block_bytes = (blocksize * b_bits).next_multiple_of(32) / 8;
    let body_bytes = correct_bytestream.take(n_block_bytes).collect_vec(); 


    let mut decoded_primary = if false {

        let mut decoded_primary2 = vec![0u32; blocksize];
        // use the bitpacking crate
        use bitpacking::{BitPacker1x, BitPacker};
        let bitpacker = BitPacker1x::new();
        
        // for whatever reason, this just unpacks 32integers!
        // we need 512!!
        // let mut buffer = vec![0_u32; 32];
        let mut pos = 0;
        for _ in 0..blocksize / 32 {

            // a view into the final vectors
            let buf = &mut decoded_primary2[pos..pos+32];

            let bytes_consumed = bitpacker.decompress(body_bytes.as_slice(), buf, b_bits as u8);
            assert_eq!(bytes_consumed, 32);
            pos+=32;
        }

        println!("test body_bytes:{} bbits: {b_bits} bytes_consumed: {}", body_bytes.len(), decoded_primary2.len());

        decoded_primary2.into_iter().map(|x| x as u64).collect_vec()

    } else {

        let packed_bits: &bv::BitSlice<MyBitStore, bv::Msb0> = bv::BitSlice::from_slice(&body_bytes);
        // Decod the primary buffer
        let mut decoded_primary: Vec<u64> = Vec::with_capacity(blocksize);
        for bits in packed_bits.chunks(b_bits) {
            decoded_primary.push(PrimaryBuffer::decode_primary_buf_element(bits));
        }
        decoded_primary
    };

    // assert_eq!(decoded_primary2.into_iter().map(|x| x as u64).collect_vec(), decoded_primary);

    // puzzle it together: at the locations of exceptions, increment the decoded values
    for (i, highest_bits) in izip!(index, exceptions) {
        let lowest_bits = decoded_primary[i as usize];
        let el = (highest_bits << b_bits) | lowest_bits;
        let pos = i as usize;
        decoded_primary[pos] = el;
    }

    //shift up againsty min_element
    let decoded_final: Vec<u64> = decoded_primary.iter().map(|x|x+min_el).collect();
    // for i in &mut decoded_primary {
    //     *i += min_el;
    // }

    (decoded_final, buf_position+n_block_bytes)
}

#[cfg(test)]
mod test {
    use fastfibonacci::{byte_decode::byte_manipulation::bits_to_fibonacci_generic_array_u32, utils::bitstream_to_string_pretty};
    use rand::prelude::Distribution;
    use rand_distr::Geometric;
    use super::*;
    #[test]
    fn test_decode(){

        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(2,  32);
        // let input = vec![1,2,3,4,5,6,7,8,9];
        let input = vec![1,2,3,4];
        let _encoded = n.encode(&input, 0);
        
        println!("{}",bitstream_to_string_pretty(&_encoded, 32));

        let bytes = bits_to_fibonacci_generic_array_u32(&_encoded);
        println!("{:?}", bytes);

        let (elements, dummy) = decode_general(&bytes, 4, 32);
        // let (elements, dummy) = decode(&bytes, 4, 32);
        println!("dummy {dummy}");
        assert_eq!(elements, input)
    }

    #[test]
    fn test_encode_decode_less_elements_than_blocksize() {
        let blocksize = 32; 
        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(4,  blocksize);
        let input = vec![0,1,2,16, 1, 17, 34, 1];
        let encoded = n.encode(&input, 0);
        let bytes = bits_to_fibonacci_generic_array_u32(&encoded);
        let (decoded,_) = decode_general(&bytes, input.len(), blocksize);
        assert_eq!(decoded, input); 
    }

    #[test]
    fn test_newpfd_codec_encode_decode_multiblock() {
        // firsst block is encodble with5 bits, second with 6
        let input: Vec<u64> = (0..50).map(|x|x).collect();
        let (encoded, _n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        let bytes = bits_to_fibonacci_generic_array_u32(&encoded);
        
        let (decoded, _) = decode_general(&bytes, input.len(), 32);
        assert_eq!(decoded, input); 
        // all the articical zero elements are now 132, since they get shifted to the min of the second block
        // assert!(decoded[input.len()..].iter().all(|&b|b==32))
    }

    #[test]
    fn test_correctness() {
        // some random data
        let n = 10_000_000;
        // let data_dist = Uniform::from(0..255);
        let data_dist = Geometric::new(0.01).unwrap();
        let mut rng = rand::thread_rng();
        let mut data: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(data_dist.sample(&mut rng));
        }

        let blocksize = 512;
        let (_enc, _) = crate::newpfd_bitvec::encode(data.iter().cloned(), blocksize);

        let bytes = bits_to_fibonacci_generic_array_u32(&_enc);

        let (decoded_data,_) = decode_general(&bytes, n, blocksize);
        assert_eq!(decoded_data, data);
    }
}