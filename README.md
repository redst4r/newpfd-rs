# NewPFD-rs

Rust library implementing the [NewPFD](https://dl.acm.org/doi/10.1145/1526709.1526764) integer compression/decompression algorithm. 

## Performance
It's currently lacking optimization for speed, but it's decently fast: 
- Encoding: 90ms/ 1M integers
- Decoding: 20ms/ 1M integers

See benchmarks for details.

### Fibonacci encoding/decoding
Fibonacci encoding/decoding is up to speed with other rust implementations, e.g. fibonnaci_codec crate (which I took some code from):
- Fibonacci encoding: 155ms/ 1M integers  (fibonnaci_codec: 160ms)
- Fibonacci decoding: 113ms/ 1M integers  (fibonnaci_codec: 140ms)

Additionally, we implemented **fast fibonacci decoding**, which is typically 2x faster than the regular fibonacci decoding:
- Fast Fibonacci decoding: 55ms/ 1M integers


## Examples
For more examples, see the rust-docs.
```rust
// Encode some data using NewPFD
use newpfd::newpfd_bitvec::{encode, decode};
let data = vec![10_u64,12,10,1,1,2,3];
let blocksize = 32; // needs to be a mutliple of 32

// encode
let (compressed_data, _) = encode(data.iter().cloned(), blocksize);
// compressed_data is a `bitvec::BitVec` (similar to a Vec<bool>)

// decode
let (decompressed_data, bits_processed) = decode(&compressed_data, data.len(), blocksize);
assert_eq!(data, decompressed_data);
assert_eq!(compressed_data.len(), bits_processed); // the entire bitstream was consumed
```
