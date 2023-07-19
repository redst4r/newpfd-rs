//! Crate implementing NewPFD (a variant of the patched ) compression of integers.
//! 
//! See [Inverted index compression and query processing with optimized document ordering](https://dl.acm.org/doi/10.1145/1526709.1526764) for the original publication.
//! 
//! # Overview
//! The algorithm splits the intput data (u64) into blocks of size `blocksize`.
//! For each block, choose a bit_width, such that 90% of the block elements can be encoded using bit_width.
//! Those elements can be stored in the primary buffer. 
//! For elements that dont fit, store the lower `bit_width` bits in the primary buffer, encode and store the excess bits separately (Fibonacci encoding)
//! 
//! # Note
//! The decoder is written in such a way that it doesn't *consume* the compressed data.
//! Rather it takes a refence to it, reads in the specific amount of elements and returns the number of bits processed.
//! 
//! This allows for complex formats where NewPFD is only a small part,
//! and after the NewPFD section, theres other data that has to be processed differently.
//! In particular, cafter alling `let (decompressed_data, bits_processed) = decode(&compressed_data, data.len(), blocksize);`
//! you can get hold of the remaining bitstream via `compressed_data[bits_processed..]`
//! 
//! # Example
//! ```rust
//! // Encode some data using NewPFD
//! # use newpfd::newpfd_bitvec::{encode, decode};
//! let data = vec![10_u64,12,10,1,1,2,3];
//! let blocksize = 32; // needs to be a mutliple of 32
//! // encode
//! let (compressed_data, _) = encode(data.iter().cloned(), blocksize);
//! // compressed_data is a `bitvec::BitVec` (similar to a Vec<bool>)
//! 
//! // decode
//! let (decompressed_data, bits_processed) = decode(&compressed_data, data.len(), blocksize);
//! assert_eq!(data, decompressed_data);
//! assert_eq!(compressed_data.len(), bits_processed); // the entire bitstream was consumed
//! ```
//! 
#![deny(missing_docs)]
pub mod newpfd_bitvec;
pub mod fibonacci;
// pub mod newpfd;
