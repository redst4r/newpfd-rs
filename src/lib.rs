//! Crate implementing NewPFD (a variant of PForDelta, aka Patched Frame Of Reference Delta) compression of integers.
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
//! * The decoder is written in such a way that it doesn't *consume* the compressed data.
//! Rather it takes a refence to it, reads in the specific amount of elements and returns the number of bits processed.
//! This allows for complex formats where NewPFD is only a small part,
//! and after the NewPFD section, theres other data that has to be processed differently.
//! In particular, after calling `let (decompressed_data, bits_processed) = decode(&compressed_data, data.len(), blocksize);`
//! you can get hold of the remaining bitstream via `compressed_data[bits_processed..]`
//! 
//! * This compression **does not** use delta compression internally. If you need that, apply it before feeding the data into `encode()`.
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
//! # Memory layout
//! 
//! Header: 
//! - fib([b_bits, min_element, #exceptions])
//! - Exceptions
//! - exception indices (delta encoded)
//! 
//! Body:
//! - |b_bits|b_bits|b_bits|b_bits|b_bits|b_bits|...|
//! 
//! # Performance
//! The library is currently *NOT* optimized for performance! 
//! I'm using some code from the fibonacci_codec crate to have efficient encoding/decoding of int-streams.
//! 
//! However, here's what we get in terms of encoding and decoding 1Mio 1byte integers (0..255):
//! ```bash,no_run     
//! mean +/- std
//! Encoding: [139.52 ms 139.99 ms 140.62 ms]
//! Decoding: [15.956 ms 16.057 ms 16.199 ms]
//! ```
//! Decoding seems to be vastly faster!
//! 
#![deny(missing_docs)]
pub mod newpfd_bitvec;
pub mod fibonacci;
pub mod fibonacci_fast;
pub mod fibonacci_old;
