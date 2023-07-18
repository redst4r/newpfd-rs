//! NewPFD with bitvec backend (instead of the old, bit_vec crate)
//! 
//! 
//! 

use bitvec::{prelude as bv, field::BitField};
// use fibonacci_codec::Encode;
use itertools::{izip, Itertools};
use crate::fibonacci::{self, bitslice_to_fibonacci, fib_enc};
// use crate::newpfd::NewPFDCodec;

/// round an integer to the next bigger multiple
/// ```rust
///  use newpfd::newpfd_bitvec::round_to_multiple;
///  assert_eq!(round_to_multiple(10,10), 10);
///  assert_eq!(round_to_multiple(11,10), 20);
///  assert_eq!(round_to_multiple(6,5), 10);
/// ```
pub fn round_to_multiple(i: usize, multiple: usize) -> usize {
    ((i+multiple-1)/multiple)*multiple
}

#[derive(Debug)]
struct NewPFDParams {
    b_bits: usize,
    min_element: u64
}

/// elements in the primary buffer are stored using b_bits
/// decode these as u64s
fn decode_primary_buf_element(x: &bv::BitSlice<u8, bv::Msb0>) -> u64 {
    let a:u64 = x.load_be();
    a
}

/// Decoding a block of NewPFD from a BitVec containing a series of blocks
/// 
/// This pops off the front of the BitVec, removing the block from the stream
/// 
/// 1. decode (Fibbonacci) metadata+expeptions+gaps
/// 2. Decode `blocksize` elements (each of sizeb_bits)
/// 3. aseemble the whole thing, filling in expecrionts etc
/// 
/// We need to know the blocksize, otherwise we'd start decoding the header of 
/// the next block
/// 
/// # NOte:
/// The bitvector x gets changed in this function. Oddly that has weird effects on this 
/// variable outside the function (before return the x.len()=9, outside the function it is suddenly 3)
/// Hence we return the remainder explicitly
fn decode_newpfdblock(buf: &bv::BitSlice<u8, bv::Msb0>, blocksize: usize) -> (Vec<u64>, usize) {

    // println!("********Start of NewPFD Block************");
    // println!("decode_newpfdblock \n{}", bitstream_to_string(buf));
    // The header, piece by piece

    let mut buf_position = 0;

    let mut fibdec = fibonacci::FibonacciDecoder::new(buf);
    let _b_bits = bitslice_to_fibonacci( fibdec.next().unwrap());
    let mut b_bits = _b_bits as usize;

    let mut min_el = bitslice_to_fibonacci( fibdec.next().unwrap());
    let mut n_exceptions = bitslice_to_fibonacci( fibdec.next().unwrap());

    b_bits -= 1;
    min_el -= 1;
    n_exceptions -= 1;
    
    // println!("Decoded Header b_bits: {b_bits} min_el: {min_el} n_exceptions: {n_exceptions}");

    // println!("Decoding gaps");
    let mut index_gaps = Vec::with_capacity(n_exceptions as usize);
    for _ in 0..n_exceptions { 
        let ix = bitslice_to_fibonacci( fibdec.next().unwrap()) - 1; // shift in encode
        index_gaps.push(ix);
    }

    // println!("Decoding exceptions");
    let mut exceptions = Vec::with_capacity(n_exceptions as usize);
    for _ in 0..n_exceptions { 
        let ex = bitslice_to_fibonacci( fibdec.next().unwrap());
        exceptions.push(ex);
    }

    let index: Vec<u64> = index_gaps
        .into_iter()
        .scan(0, |acc, i| {
            *acc += i;
            Some(*acc)
        })
        .collect();

    // println!("remain {:?}, len {}", x, x.len(), );
    // need to remove trailing 0s which where used to pad to a muitple of 32
    // let bits_after = x.len();
    // let delta_bits  = bits_before-bits_after;
    let delta_bits = fibdec.get_bits_processed();
    let padded_bits =  round_to_multiple(delta_bits, 32) - delta_bits;

    assert!(!buf[buf_position+delta_bits..buf_position + delta_bits + padded_bits].any());
    buf_position = buf_position + delta_bits + padded_bits;


    // the body of the block
    let buf_body = &buf[buf_position..];
    let mut body_pos = 0;
    // println!("buf_body \n{}", bitstream_to_string(buf_body));
    
    // note that a block can be shorter than sepcifiec if we ran out of elements
    // however, as currently implements (by bustools)
    // the primary block always has blocksize*b_bits bitsize!
    // i.e. for a partial block, we cant really know how many elements are in there
    // (they might all be 0)
    /*
    let n_elements = if (x.len() /b_bits) < blocksize {
        assert_eq!(x.len() % b_bits, 0); 
        let n_elem_truncate = x.len() / b_bits;
        n_elem_truncate
    } else {
        blocksize
    };
    */
    let n_elements = blocksize;
    let mut decoded_primary: Vec<u64> = Vec::with_capacity(n_elements);

    for _ in 0..n_elements {
        // println!("++++++++++++++Element {}/{}++++++++++++*", i+1, n_elements);
        // println!("{:?}", x);
        // split off an element into x
        let bits = &buf_body[body_pos..body_pos+b_bits];
        body_pos+=b_bits;
        // println!("buf_pos {}",         body_pos);
        // println!("n_El {}",         i);
        // println!("prim bits {}", bitstream_to_string(bits));

        decoded_primary.push(decode_primary_buf_element(bits));
        // println!("remaining size {}", x.len())
    }
    // println!("********End of NEWPFD  Block************\n");
    // println!("decode Index: {:?}", index);
    // println!("decode Excp: {:?}", exceptions);
    // println!("decode prim: {:?}", decoded_primary);
    // println!("decode min: {}", min_el);

    // puzzle it together
    for (i, highest_bits) in izip!(index, exceptions) {
        let lowest_bits = decoded_primary[i as usize];
        let el = (highest_bits << b_bits) | lowest_bits;
        let pos = i as usize;
        decoded_primary[pos] = el;
    }

    //shift up againsty min_element
    let decoded_final: Vec<u64> = decoded_primary.iter().map(|x|x+min_el).collect();

    // println!("!!!remaining size {}, {:?}", x.len(), x);
    // println!("Final decode {:?}", decoded_final);
    (decoded_final, buf_position+body_pos)
}

/// Decode a NewPFD-encoded buffer, containing `n_elements`
pub fn decode(newpfd_buf: &bv::BitSlice<u8, bv::Msb0>, n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){

    let mut pos = 0;
    let mut elements: Vec<u64> = Vec::with_capacity(n_elements);
    while elements.len() < n_elements {
        // println!("Decoding newPFD Block {}: {:?}, len={}", i, encoded, encoded.len());
        // each call shortens wth encoded BitVec

        let current_block = &newpfd_buf[pos..];
        let (els, bits_consumed) = decode_newpfdblock(current_block, blocksize);

        pos+= bits_consumed;
        // println!("----remaining size {}, {:?}", encoded.len(), encoded);
        // println!("Decoded {} elements", els.len());

        for el in els {
            elements.push(el);
        }
    }
    // trucate, as we retrieved a bunch of zeros from the last block
    elements.truncate(n_elements);

    (elements, pos)
}

/// just for debuffing purpose
fn bitstream_to_string(buffer: &bv::BitSlice<u8, bv::Msb0>) -> String{
    let s = buffer.iter().map(|x| if *x==true{"1"} else {"0"}).join("");
    s
}

/// Data Stored in a single block of the NewPFD format
/// 
/// # Format
/// Each block starts with a header:
/// * b_bits
/// * minimum element
/// * number of exceptiion
/// * delta encoded index of exceptions
/// * exceptations
/// 
/// The body is a section of `blocksize * b_bits` (the primary buffer), 
/// where each element is encoded using `b_bits` in the primary_buffer. 
/// Elements not fitting into `b_bits` are stored as exceptions
/// * the `b_bits` lower bits of an exception go into the primary buffer.
/// * the higher bits are stored after the primary buffer
struct NewPFDBlock {
    // blocksize: usize,
    b_bits: usize,  // The number of bits each num in `pfd_block` is represented with.
    blocksize: usize, //usually 512, needs to be a multiple of 32
    primary_buffer: Vec<bv::BitVec<u8, bv::Msb0>>,
    exceptions: Vec<u64>,
    index_gaps : Vec<u64>
}

impl NewPFDBlock {
    pub fn new(b_bits: usize, blocksize: usize) -> Self {

        assert_eq!(blocksize % 32, 0, "Blocksize must be mutiple of 32");
        let pb = Vec::with_capacity(blocksize);
        let exceptions: Vec<u64> = Vec::new();
        let index_gaps : Vec<u64> = Vec::new();
        NewPFDBlock { 
            // blocksize: blocksize, 
            b_bits: b_bits, 
            blocksize,
            primary_buffer: pb, 
            exceptions: exceptions, 
            index_gaps: index_gaps 
        }
    }

    /// determine the bitwidth used to store the elements of the block
    /// and the minimum element of the block 
    pub fn get_encoding_params(input: &[u64], percent_threshold: f64) -> NewPFDParams {

        // get the `percent_threshold` quantile of the input stream
        let mut percent_ix = (input.len() as f64 ) * percent_threshold;
        percent_ix -= 1.0; // e..g if len = 4, percent=0.75  percent_ix should be pointing to the 3rd element, ix=2
        let ix = percent_ix.round() as usize;
        
        let (minimum, mut n_bits) = if ix == 0 {
            let the_element = input[0];
            (the_element, u64::BITS - the_element.leading_zeros())
        } else {
            let mut sorted = input.iter().sorted();
            let minimum = *sorted.next().unwrap();
            let mut the_element = *sorted.nth(ix-1).unwrap();   //-1 since we took the first el out 
            the_element -= minimum; // since we're encoding relative to the minimum
            let n_bits = u64::BITS - the_element.leading_zeros(); 
            (minimum, n_bits)
        };

        // weird exception: IF there's only a single bit to encode (blocksize==1)
        // and that bit happens to be zero -> n_bits==0 which will cause problems down the road
        // hence set n_bits>=1
        if n_bits == 0 {
            n_bits = 1;
        } 
        NewPFDParams { b_bits: n_bits as usize, min_element: minimum}
    }

    /// turns input data into the NewPFD storage format:
    /// represent items by b_bits, store any expections separately 
    // pub fn from_data(input: &[u64], b_bits: usize) -> Self{
    pub fn encode(&mut self, input: &[u64], min_element: u64) -> bv::BitVec<u8, bv::Msb0> {

        assert!(input.len() <= self.blocksize);

        // this is the maximum element we can store using b_bits
        let max_elem_bit_mask = (1_u64 << self.b_bits) - 1;

        // index of the last exception we came across
        let mut last_ex_idx = 0;

        // go over all elements, split each element into low bits (fitting into the primary)
        // and the high bits, which go into the exceptions
        for (i,x) in input.iter().enumerate() {
            let mut diff = x- min_element; // all elements are stored relative to the min

            // if its an exception, ie to big to be stored in b-bits
            // we store the overflowing bits seperately
            // and remember where this exception is
            // todo: convert diff to BitSlice, subset the bits into head|tail
            if diff > max_elem_bit_mask {
                // println!("Exception {}", diff);
                self.exceptions.push(diff >> self.b_bits);
                self.index_gaps.push((i as u64)- last_ex_idx);
                last_ex_idx = i as u64;

                // write the stuff that fits into the field
                diff &= max_elem_bit_mask;  // bitwise and with 111111...1111 to get the `b_bits` least significant bits
            }

            // put the rest into the primary buffer
            // turn the element to be stored into a bitvec (it'll be bigger than intended, but we'll chop it)
            // BigEndian: least significant BYTES are on the right!

            let bvec = &bv::BitVec::from_slice(&diff.to_be_bytes());
            let (zero, cvec) = bvec.split_at(bvec.len()-self.b_bits);
            
            assert_eq!(cvec.len(), self.b_bits);
            assert!(!zero.any()); // the chopped off part should be 0
            // println!("{:?}", cvec);
            self.primary_buffer.push(cvec.to_bitvec());
        }
        // println!("encode: Primary {:?}", self.primary_buffer);
        // println!("encode: b_bits {}", self.b_bits);
        // println!("encode: min_el {}", min_element);
        // println!("encode: n_ex {}", self.exceptions.len());
        // println!("encode: Exceptions {:?}", self.exceptions);
        // println!("encode: Gaps: {:?}", self.index_gaps);

        // merge the primary buffer into a single bitvec, the body of the block
        let mut body: bv::BitVec<u8, bv::Msb0> = bv::BitVec::with_capacity(self.b_bits+self.blocksize); // note that we need to pad with later

        for mut b in self.primary_buffer.iter_mut() { 
            body.append(&mut b);
        }

        // the primary buffer must fit into a multiple of u32!
        // if the block is fully occupied (512 el) this is naturally the case
        // but for partial blocks, we need to pad
        // Actually, bustools implements it as such that primary buf is ALWAYS
        // 512 * b_bits in size (no matter how many elements)
        let total_size = self.blocksize * self.b_bits;
        assert_eq!(total_size % 32, 0); 
        let to_pad = total_size - body.len();
        for _ in 0..to_pad {
            body.push(false);
        }
        // println!("padded Body by {} bits to {}", to_pad, body.len());
        // println!("{:?}", body);

        // now, put the "BlockHeader"  in front of the merged_primary buffer we store via fibonacci encoding:
        // 1. b_bits
        // 2. min_element
        // 3.n_exceptions
        // 4. index_gaps
        // 5. exceptions
        let mut to_encode: Vec<u64> = vec![
            1 + self.b_bits as u64, 
            1 + min_element, 
            1 + self.exceptions.len() as u64
        ];
        // adding the exceptions
        to_encode.extend(self.index_gaps.iter().map(|x| x+1)); //shifting the gaps +1
        to_encode.extend(self.exceptions.iter());

        let mut header = bv::BitVec::new();
        for el in to_encode {
            let mut f = fib_enc(el);
            header.append(&mut f);
        }

        // yet again, this needs to be a mutliple of 32
        let to_pad =  round_to_multiple(header.len(), 32) - header.len();
        for _ in 0..to_pad {
            header.push(false);
        }
        // println!("padded Header by {} bits to {}", to_pad, header.len());

        // merge header + body
        header.extend(body);
        header
    }

}

/// NewPDF encoding 
/// # Example
/// ```rust
/// 
/// ```
pub struct NewPFDCodec {
    blocksize: usize,
}

impl NewPFDCodec {

    /// initialize NewPDF encoding with blocksize (usually 512)
    pub fn new(blocksize: usize) -> Self {
        NewPFDCodec { blocksize }
    }

    /// encode a stream of u64s with NewPFD, returning a bitvector
    pub fn encode(&self, input_stream: impl Iterator<Item=u64>) -> bv::BitVec<u8, bv::Msb0> {

        let mut encoded_blocks : Vec<bv::BitVec<u8, bv::Msb0>> = Vec::new();
        // chunk the input into blocks
        for b in &input_stream.chunks(self.blocksize) {

            let block_elements: Vec<_> = b.collect();

            let params = NewPFDBlock::get_encoding_params(&block_elements, 0.9);
            // println!("newpfd params: {params:?}");
            let mut enc = NewPFDBlock::new(params.b_bits, self.blocksize);
            let enc_block = enc.encode(&block_elements, params.min_element as u64);
            encoded_blocks.push(enc_block);
        }
        // concat
        let mut iii = encoded_blocks.into_iter();
        let mut merged_blocks = iii.next().unwrap();
        for mut b in iii {
            merged_blocks.append(&mut b);
        }
        merged_blocks
    }

    /// decode the newpfd encoding
    /// due to the zero padding of truncated blocks, the format itself has no info
    /// about how many elements are stored. Hence `n_elements` needs to be supplied
    /// 
    /// It'll always decode an entire block before checking if we have enough elements
    pub fn decode(&self, encoded: bv::BitVec<u8, bv::Msb0>, n_elements: usize) -> (Vec<u64>, bv::BitVec<u8, bv::Msb0> ){
        let mut elements: Vec<u64> = Vec::with_capacity(n_elements);
        let mut i = 0;
        let mut encoded_pos = 0;
        while elements.len() < n_elements {
            // println!("Decoding newPFD Block {}: {:?}, len={}", i, encoded, encoded.len());
            i+=1;
            // each call shortens wth encoded BitVec
            let (els, bits_processed) = decode_newpfdblock(&encoded[encoded_pos..], self.blocksize);
            encoded_pos += bits_processed;
            // println!("----remaining size {}, {:?}", encoded.len(), encoded);
            // println!("Decoded {} elements", els.len());

            for el in els {
                elements.push(el);
            }
        }
        // trucate, as we retrieved a bunch of zeros from the last block
        elements.truncate(n_elements);

        (elements,encoded)
    }
}


#[cfg(test)]
mod test {
    use bitvec::prelude as bv;
    use super::{decode, NewPFDBlock};
    #[test]
    fn test_larger_ecs_22() {
        let input = vec![264597, 760881, 216982, 203942, 218976];
        
        let n = crate::newpfd_bitvec::NewPFDCodec::new(32);

        // println!("{:?}", NewPFDBlock::get_encoding_params(&input, 0.9));
        let encoded = n.encode(input.iter().cloned());
        // println!("Encoded:\n{:?}", encoded);
        let encoded_bv: bv::BitVec<u8, bv::Msb0> = bv::BitVec::from_iter(encoded.iter());
        // println!("Encoded bv:\n{}", bitstream_to_string(&encoded_bv));

        let (decoded, _) = decode(&encoded_bv.as_bitslice(), 5, 32);
        
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_padding() {
        // each block (header+body)must be a mutiple of u32
        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(4,  32);
        let input = vec![0,1,2,16, 1, 17];
        let b = n.encode(&input, 0);
        assert_eq!(b.len() % 32, 0);
        // assert_eq!(b.len(), n.blocksize * n.b_bits); // only applies to body
    }

    #[test]
    fn test_params() {
        let input = vec![0,10, 100, 1000];
        let params = crate::newpfd_bitvec::NewPFDBlock::get_encoding_params(&input, 0.75);
        assert_eq!(params.min_element, 0);
        assert_eq!(params.b_bits, 7);
    }

    #[test]
    fn test_params_min_el() {
        let input = vec![1,10, 100, 1000];
        let params = crate::newpfd_bitvec::NewPFDBlock::get_encoding_params(&input, 0.75);
        assert_eq!(params.min_element, 1);
        assert_eq!(params.b_bits, 7);
    }


    #[test]
    fn test_encode_decode_less_elements_than_blocksize() {
        let blocksize = 32; 
        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(4,  blocksize);
        let input = vec![0,1,2,16, 1, 17, 34, 1];
        let mut encoded = n.encode(&input, 0);


        // println!("Enc length {}", encoded.len());
        // println!("Plain length {}", input.len()*64);

        let encoded_bv: bv::BitVec<u8, bv::Msb0> = bv::BitVec::from_iter(encoded.iter());

        let (decoded,_) = crate::newpfd_bitvec::decode_newpfdblock(encoded_bv.as_bitslice(), blocksize);

        //problem here: we dont know how many elements in a truncated block
        // the block will always be filled up with zero elements to get to `blocksize` elements
        assert_eq!(decoded[..input.len()], input); 
    }
    #[test]
    fn test_encode_no_exceptions() {
        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(2,  32);
        let input = vec![0,1,0, 1];
        let _encoded = n.encode(&input, 0);

        assert_eq!(n.exceptions.len(), 0);
        assert_eq!(n.index_gaps.len(), 0);
    }

    #[test]
    fn test_newpfd_codec_encode_decode() {
        let n = crate::newpfd_bitvec::NewPFDCodec::new(32);
        let input = vec![0_u64,1,0, 1];
        let encoded = n.encode(input.iter().cloned());
        let (decoded,_) = n.decode(encoded, input.len());

        //same problem as above: truncated blocks contain trailing 0 eleemnts
        assert_eq!(decoded, input);
    }


    // #[test]
    // not relevant any more, forxing blocksize as a multipel of 32
    fn test_newpfd_codec_encode_decode_blocksize1() {
        // blocksize==1 exposes some edge cases, like a single 0bit in the block
        let n = crate::newpfd_bitvec::NewPFDCodec::new(1);
        let input = vec![0_u64,1,0, 1];
        let encoded = n.encode(input.iter().cloned());

        let (decoded,_) = n.decode(encoded, input.len());
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_newpfd_codec_encode_decode_nonzero_min_el() {
        let n = crate::newpfd_bitvec::NewPFDCodec::new(32);
        let input = vec![1_u64,2,2, 1];
        let encoded = n.encode(input.iter().cloned());
        let (decoded,_) = n.decode(encoded, input.len());
        assert_eq!(decoded[..input.len()], input);

        // all the articical zero elements are now 1, since they get shifted
        assert!(decoded[input.len()..].iter().all(|&b|b==1))
    }
    #[test]
    fn test_newpfd_codec_encode_decode_multiblock() {
        // firsst block is encodble with5 bits, second with 6
        let n = crate::newpfd_bitvec::NewPFDCodec::new(32);
        let input: Vec<u64> = (0..50).map(|x|x).collect();
        let encoded = n.encode(input.iter().cloned());
        let (decoded, _) = n.decode(encoded, input.len());
        
        assert_eq!(decoded, input);

        // all the articical zero elements are now 132, since they get shifted to the min of the second block
        assert!(decoded[input.len()..].iter().all(|&b|b==32))
    }

    #[test]
    fn test_larger_ecs() {
        let input = vec![264597, 760881, 216982, 203942, 218976];
        // let input = vec![1,2,3,4,5];
        let n = crate::newpfd_bitvec::NewPFDCodec::new(32);

        println!("{:?}", crate::newpfd_bitvec::NewPFDBlock::get_encoding_params(&input, 0.9));
        let encoded = n.encode(input.iter().cloned());
        let (decoded, _) = n.decode(encoded, input.len());
        
        assert_eq!(decoded, input);
    }


    #[test]
    fn test_param() {
        let x = NewPFDBlock::get_encoding_params(&[10_u64, 10,10,10], 0.9);
        assert_eq!(x.b_bits, 1);
        assert_eq!(x.min_element, 10);
    }
}