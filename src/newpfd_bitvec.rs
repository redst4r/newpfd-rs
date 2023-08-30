//! NewPFD with bitvec backend (instead of the old, bit_vec crate)
//! 

use bitvec::{prelude as bv, field::BitField};
use itertools::{izip, Itertools};
use crate::fibonacci::{self, fib_enc_multiple_fast};

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

/// Decode a NewPFD-encoded buffer, containing `n_elements`. 
/// Due to limitations of the format, we can't know (internally) how many elements were stored, 
/// hence `n_elements` needs to be specified
/// 
/// # Parameters
/// * `newpfd_buf`: A BitSlice containing NewPFD encoded data
/// * `n_elements`: number of elements to decode
/// * `blocksize`: Blocksize used to encode the buffer
/// # Returns:
/// * A `Vec<u64>` of decoded values (len()==n_elements)
/// * the number of bits that were processed in the input buffer. 
///   Useful if one wants to use the buffer afterwards (in case theres other data in there)
/// # Example
/// ```rust
/// # use bitvec::prelude as bv;
/// # use newpfd::newpfd_bitvec::{encode, decode};
/// let data = vec![1,2,3,4,5];
/// let blocksize = 32;
/// let (encoded, n_elements) = encode(data.iter().cloned(), 32);
/// 
/// let (decoded, bits_processed) = decode(&encoded, n_elements, blocksize);
///
/// // other data (if there were any)
/// let remaining_buffer = &encoded[bits_processed..];
/// 
/// assert_eq!(data, decoded);
/// assert_eq!(encoded.len(), bits_processed);
/// ``` 
/// 
pub fn decode(newpfd_buf: &bv::BitSlice<u8, bv::Msb0>, n_elements: usize, blocksize: usize) -> (Vec<u64>, usize){

    let mut pos = 0;
    let mut elements: Vec<u64> = Vec::with_capacity(n_elements);
    while elements.len() < n_elements {
        // each call shortens the encoded BitVec
        let current_block = &newpfd_buf[pos..];
        let (els, bits_consumed) = decode_newpfdblock(current_block, blocksize);

        pos+= bits_consumed;

        for el in els {
            elements.push(el);
        }
    }
    // trucate, as we retrieved a bunch of zeros from the last block
    elements.truncate(n_elements);

    (elements, pos)
}

/// encode data using NewPFD, 
/// 
/// # Parameters
/// * `input_stream`: Any iterator yielding u64
/// * `blocksize`: Number of elements going into a single NewPFD block (which gets compressed with b_bits and exceptions).
/// Must be a mutliple of 32!
/// 
/// # Returns
/// * a BitVec containing the NewPFD encoded data
/// * the number of elements that were encoded (size of the input iterator)
/// # Example
/// ```rust
/// # use bitvec::prelude as bv;
/// # use newpfd::newpfd_bitvec::encode;
/// let data = vec![1,2,3,4,5];
/// let (encoded, _) = encode(data.into_iter(), 32);
/// ``` 
pub fn encode(input_stream: impl Iterator<Item=u64>, blocksize: usize) -> (bv::BitVec<u8, bv::Msb0>, usize) {

    let mut n_elements = 0;
    let mut encoded_blocks : Vec<bv::BitVec<u8, bv::Msb0>> = Vec::new();
    // chunk the input into blocks
    for b in &input_stream.chunks(blocksize) {

        let block_elements: Vec<_> = b.collect();
        n_elements +=  block_elements.len();

        let params = NewPFDBlock::get_encoding_params(&block_elements, 0.9);
        let mut enc = NewPFDBlock::new(params.b_bits, blocksize);
        let enc_block = enc.encode(&block_elements, params.min_element);
        encoded_blocks.push(enc_block);
    }
    // concat
    let mut iii = encoded_blocks.into_iter();
    let mut merged_blocks = iii.next().unwrap();
    for mut b in iii {
        merged_blocks.append(&mut b);
    }
    (merged_blocks, n_elements)
}


// /// just for debugging purpose
// fn bitstream_to_string(buffer: &bv::BitSlice<u8, bv::Msb0>) -> String{
//     let s = buffer.iter().map(|x| if *x{"1"} else {"0"}).join("");
//     s
// }

/// Primary Buffer of the NewPFD block
/// 
/// Encodes each element with `b_bits`, bitpacking everything together
/// **Warning** this DOES NOT check if each element fits into `b_bits`
/// but just takes each elements lowerest `b_bits`, truncating the higher bits
#[derive(Debug)]
struct PrimaryBuffer {
    buffer: bv::BitVec<u8, bv::Msb0>,
    b_bits: usize,     // bits per int
    blocksize: usize,  // max number of elements that can be stored
    position: usize,   // current bitposition in the buffer
    max_elem_bit_mask : u64  // max element encodable by b_bits
}

impl PrimaryBuffer {
    /// create an empty primary buffer, storing `blocksize` elements, using `b_bits` Bits per element
    pub fn new(b_bits: usize, blocksize: usize) -> Self {
        let buffer = bv::BitVec::repeat(false, b_bits*blocksize);

        // this is the maximum element we can store using b_bits
        let max_elem_bit_mask = (1_u64 << b_bits) - 1;

        PrimaryBuffer {
            buffer,
            b_bits,
            blocksize,
            position: 0,
            max_elem_bit_mask
        }
    }

    fn get_n_elements(&self) -> usize{
        self.position / self.b_bits
    }

    /// adds a single element to the primary buffer, storing its lowest `b_b its`
    /// if there's excess bits, they are returned as Some(u64) otherwise none 
    pub fn add_element(&mut self, el: u64) -> Option<u64>{

        if self.get_n_elements() >= self.blocksize {
            panic!("storing too many elements")
        }

        self.buffer[self.position..self.position+self.b_bits].store_be::<u64>(el); //store_be chops of higher bits
        self.position+=self.b_bits;

        // any excess bits that didnt fit?
        // TODO: nicer synthax: convert diff to BitSlice, subset the bits into head|tail
        if el > self.max_elem_bit_mask {
            let excess =el >> self.b_bits;
            Some(excess)
        }
        else {
            None
        }
    }

    /// decodes the ENTIRE buffer, even if not fully filled
    /// trailing elements will be zero 
    pub fn decode(&self) -> Vec<u64> {

        let mut decoded_primary: Vec<u64> = Vec::with_capacity(self.blocksize);
        let mut pos = 0;
        for _ in 0..self.blocksize {
            // split off an element into x
            let bits = &self.buffer[pos..pos+self.b_bits];
            pos+=self.b_bits;
            decoded_primary.push(decode_primary_buf_element(bits));
        }
        decoded_primary
    }

    // problem: ned to move buffer, but usually its only a view
    pub fn from_bitvec(b: bv::BitVec<u8, bv::Msb0>, b_bits:usize) -> Self {

        assert_eq!(b.len() % b_bits, 0, "buffer length is not a multiple of bitsize");
        let blocksize = b.len() / b_bits;
        let nbits: usize = b.len();
        let max_elem_bit_mask = (1_u64 << b_bits) - 1;

        PrimaryBuffer { buffer: b, b_bits, blocksize, position: nbits, max_elem_bit_mask }
    }
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
    b_bits: usize,  // The number of bits each num in `pfd_block` is represented with.
    blocksize: usize, // the max number of elements to be stored; usually 512, needs to be a multiple of 32
    // primary_buffer: Vec<bv::BitVec<u8, bv::Msb0>>,
    exceptions: Vec<u64>,
    index_gaps : Vec<u64>
}

impl NewPFDBlock {
    /// create NewPDFBlock, with bbits per interger, and a total number of elements == blocksize
    fn new(b_bits: usize, blocksize: usize) -> Self {

        assert_eq!(blocksize % 32, 0, "Blocksize must be mutiple of 32");
        // let pb = Vec::with_capacity(blocksize);
        let exceptions: Vec<u64> = Vec::new();
        let index_gaps : Vec<u64> = Vec::new();
        NewPFDBlock { 
            b_bits, 
            blocksize,
            // primary_buffer: pb, 
            exceptions, 
            index_gaps 
        }
    }

    /// determine the bitwidth used to store the elements of the block
    /// and the minimum element of the block
    /// Bitwidth is chosen such that `percent_threshold * input.len()` elements
    /// can be encoded by `b_bits`, typically 0.9 (90%)
    pub fn get_encoding_params(input: &[u64], percent_threshold: f64) -> NewPFDParams {

        // get the `percent_threshold` quantile of the input stream
        let mut percent_ix = (input.len() as f64 ) * percent_threshold;
        percent_ix -= 1.0; // e..g if len = 4, percent=0.75  percent_ix should be pointing to the 3rd element, ix=2
        let ix = percent_ix.round() as usize;
        
        let (minimum, mut n_bits) = if ix == 0 {
            let the_element = input[0];
            (the_element, u64::BITS - the_element.leading_zeros())
        } else {
            // TODO: no need to fully sort! partial sort would be ok too
            // let (_, the_element, _) = input.select_nth_unstable(ix);

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
    pub fn encode(&mut self, input: &[u64], min_element: u64) -> bv::BitVec<u8, bv::Msb0> {

        assert!(input.len() <= self.blocksize);

        // index of the last exception we came across
        let mut last_ex_idx = 0;

        let mut prim_buf = PrimaryBuffer::new(self.b_bits, self.blocksize);
        // go over all elements, split each element into low bits (fitting into the primary)
        // and the high bits, which go into the exceptions
        for (i,x) in input.iter().enumerate() {
            let diff = x- min_element; // all elements are stored relative to the min

            // add the element to primary buffer and see if theres any excess
            let any_excess_bits = prim_buf.add_element(diff);
            
            // if its an exception, ie to big to be stored in b-bits
            // we store the overflowing bits seperately
            // and remember where this exception is            
            if let Some(ebits) = any_excess_bits {
                self.exceptions.push(ebits);
                self.index_gaps.push((i as u64)- last_ex_idx);
                last_ex_idx = i as u64;                
            }
        }

        let body = prim_buf.buffer;

        // the primary buffer must fit into a multiple of u32! the format demands it
        // TODO this is already true for the prim_buf implementation
        assert_eq!(body.len() % 32, 0); 

        // now, put the "BlockHeader"  in front of the merged_primary buffer we store via fibonacci encoding:
        // 1. b_bits
        // 2. min_element
        // 3. n_exceptions
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

        let mut header= fib_enc_multiple_fast(&to_encode);

        // yet again, this needs to be a mutliple of 32
        let to_pad =  round_to_multiple(header.len(), 32) - header.len();
        for _ in 0..to_pad {
            header.push(false);
        }

        // merge header + body
        header.extend(body);
        header
    }

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

    let mut buf_position = 0;
    let mut fibdec = fibonacci::FibonacciDecoder::new(buf);
    // pulling the elements out of the header (b_bits, min_el, n_exceptions)
    let _b_bits = fibdec.next().unwrap();
    let mut b_bits = _b_bits as usize;
    let mut min_el = fibdec.next().unwrap();
    let mut n_exceptions = fibdec.next().unwrap();

    // as fibonaccis encoding doesnt allow 0 everything is shifted by +1
    // undo that here
    b_bits -= 1;
    min_el -= 1;
    n_exceptions -= 1;

    // decoding the Gaps
    let mut index_gaps = Vec::with_capacity(n_exceptions as usize);
    for _ in 0..n_exceptions { 
        let ix =  fibdec.next().unwrap() - 1; // shift in encode
        index_gaps.push(ix);
    }

    // Decoding expections
    let mut exceptions = Vec::with_capacity(n_exceptions as usize);
    for _ in 0..n_exceptions { 
        let ex = fibdec.next().unwrap();
        exceptions.push(ex);
    }

    // turn index gaps into the actual index (of expections)
    let index: Vec<u64> = index_gaps
        .into_iter()
        .scan(0, |acc, i| {
            *acc += i;
            Some(*acc)
        })
        .collect();

    // need to remove trailing 0s which where used to pad to a muitple of 32
    let delta_bits = fibdec.get_bits_processed();
    let padded_bits =  round_to_multiple(delta_bits, 32) - delta_bits;
    assert!(!buf[buf_position+delta_bits..buf_position + delta_bits + padded_bits].any());
    
    // move the new buffer position at the end
    buf_position = buf_position + delta_bits + padded_bits;

    // the body of the block, i.e. bitpacked integers
    let buf_body = &buf[buf_position..];

    // let buf_body_full = &buf[buf_position..buf_position+(blocksize*b_bits)];

    let mut body_pos = 0;
    
    // note that a block can be shorter than sepcifiec if we ran out of elements
    // however, as currently implements (by bustools)
    // the primary block always has blocksize*b_bits bitsize!
    // i.e. for a partial block, we cant really know how many elements are in there
    // (they might all be 0)

    let n_elements = blocksize;
    let mut decoded_primary: Vec<u64> = Vec::with_capacity(n_elements);

    // TODO: move this code into PrimaryBuffer
    for _ in 0..n_elements {
        // split off an element into x
        let bits = &buf_body[body_pos..body_pos+b_bits];
        body_pos+=b_bits;
        decoded_primary.push(decode_primary_buf_element(bits));
    }

    // puzzle it together: at the locations of exceptions, increment the decoded values
    for (i, highest_bits) in izip!(index, exceptions) {
        let lowest_bits = decoded_primary[i as usize];
        let el = (highest_bits << b_bits) | lowest_bits;
        let pos = i as usize;
        decoded_primary[pos] = el;
    }

    //shift up againsty min_element
    let decoded_final: Vec<u64> = decoded_primary.iter().map(|x|x+min_el).collect();
    (decoded_final, buf_position+body_pos)
}


#[cfg(test)]
mod test {
    use bitvec::prelude as bv;
    use rand::{distributions::Uniform, prelude::Distribution};
    use super::{decode, encode};
    #[test]
    fn test_larger_ecs_22() {
        let input = vec![264597, 760881, 216982, 203942, 218976];
        let (encoded, n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        assert_eq!(n_el, input.len());
        let encoded_bv: bv::BitVec<u8, bv::Msb0> = bv::BitVec::from_iter(encoded.iter());
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

    mod params {
        use crate::newpfd_bitvec::NewPFDBlock;
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
        fn test_param() {
            let x = NewPFDBlock::get_encoding_params(&[10_u64, 10,10,10], 0.9);
            assert_eq!(x.b_bits, 1);
            assert_eq!(x.min_element, 10);
        }
    }
 
    #[test]
    fn test_encode_decode_less_elements_than_blocksize() {
        let blocksize = 32; 
        let mut n = crate::newpfd_bitvec::NewPFDBlock::new(4,  blocksize);
        let input = vec![0,1,2,16, 1, 17, 34, 1];
        let encoded = n.encode(&input, 0);

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
        let input = vec![0_u64,1,0, 1];
        let (encoded, n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        assert_eq!(n_el, input.len());

        let (decoded,_) = decode(&encoded, input.len(), 32);
        //same problem as above: truncated blocks contain trailing 0 eleemnts
        assert_eq!(decoded, input);
    }

    // #[test]
    // not relevant any more, forxing blocksize as a multipel of 32
    #[allow(dead_code)]
    fn test_newpfd_codec_encode_decode_blocksize1() {
        // blocksize==1 exposes some edge cases, like a single 0bit in the block
        let input = vec![0_u64,1,0, 1];
        let (encoded, _n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        let (decoded,_) = decode(&encoded, input.len(), 32);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_newpfd_codec_encode_decode_nonzero_min_el() {
        let input = vec![1_u64,2,2, 1];
        let (encoded, _n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        let (decoded,_) = decode(&encoded, input.len(), 32);
        assert_eq!(decoded[..input.len()], input);
        // all the articical zero elements are now 1, since they get shifted
        assert!(decoded[input.len()..].iter().all(|&b|b==1))
    }
    #[test]
    fn test_newpfd_codec_encode_decode_multiblock() {
        // firsst block is encodble with5 bits, second with 6
        let input: Vec<u64> = (0..50).map(|x|x).collect();
        let (encoded, _n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        let (decoded, _) = decode(&encoded, input.len(), 32);
        assert_eq!(decoded, input);
        // all the articical zero elements are now 132, since they get shifted to the min of the second block
        assert!(decoded[input.len()..].iter().all(|&b|b==32))
    }

    #[test]
    fn test_larger_ecs() {
        let input = vec![264597, 760881, 216982, 203942, 218976];
        println!("{:?}", crate::newpfd_bitvec::NewPFDBlock::get_encoding_params(&input, 0.9));
        let (encoded, _n_el) = crate::newpfd_bitvec::encode(input.iter().cloned(), 32);
        let (decoded, _) = decode(&encoded, input.len(), 32);
        assert_eq!(decoded, input);
    }


    #[test]
    fn test_encode_speed() {
        // some random data
        let n = 1_000_000;
        let data_dist = Uniform::from(0..255);
        let mut rng = rand::thread_rng();
        let mut data: Vec<u64> = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(data_dist.sample(&mut rng));
        }

        let blocksize = 512;
        let _enc = encode(data.into_iter(), blocksize);
    }

    mod test_primary {
        use crate::newpfd_bitvec::PrimaryBuffer;
        #[test]
        fn test_primary() {
            let mut p = PrimaryBuffer::new(2, 3);
            p.add_element(3); // 11 in binary
            p.add_element(0); // 00 in binary
            p.add_element(3); // 11 in binary
            assert_eq!(
                p.buffer.iter().collect::<Vec<_>>(), 
                vec![true, true, false, false, true, true] 
            );
        }
        #[test]
        fn test_primary_overflow() {
            let mut p = PrimaryBuffer::new(2, 3);
            p.add_element(5); // 101 in binary

            // should onyl store the lower two bits
            assert_eq!(
                p.buffer.iter().collect::<Vec<_>>(), 
                vec![false, true, false, false, false, false] 
            );
        }

        #[test]
        #[should_panic(expected = "storing too many elements")]
        fn test_primary_too_many_el() {
            let mut p = PrimaryBuffer::new(2, 2);
            p.add_element(1); 
            p.add_element(1); 
            p.add_element(1); // excess element
        }

        #[test]
        fn test_n_elements() {
            let mut b = PrimaryBuffer::new(3, 512);
            b.add_element(10);
            assert_eq!(b.get_n_elements(), 1);
            b.add_element(0);
            assert_eq!(b.get_n_elements(), 2);
        }

        #[test]
        fn test_encode_decode_no_overflow() {
            // in 3 bits we can encode anything [0,7]
            let v = vec![0, 1,2,3,4,5,6,7];
            let mut b = PrimaryBuffer::new(3, 32);
            for el in v.iter() {
                b.add_element(*el);
            }
            let dec = b.decode();
            assert_eq!(dec.len(), 32);
            assert_eq!(dec[..v.len()], v);
        }
        #[test]
        fn test_encode_decode_with_overflow() {
            // in 2 bits we can encode anything [0,4]
            let v = vec![0,1,2,3,4,5,6,7];
            let mut b = PrimaryBuffer::new(2, 32);
            for el in v.iter() {
                b.add_element(*el);
            }
            let dec = b.decode();
            assert_eq!(dec.len(), 32);
            assert_eq!(dec[..v.len()], vec![0,1,2,3,0,1,2,3]);
        }
    }
}