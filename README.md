# NewPFD-rs

Rust library implementing the [NewPFD](https://dl.acm.org/doi/10.1145/1526709.1526764) integer compression/decompression algorithm. 

## Performance
It's currently lacking optimization for speed, but it's decently fast.
We perform this on geometrically distributed integers (Geo(lambda=0.01)) to force encoding exceptions in the NewPFD-block.
- Encoding: 90ms/ 1M integers
- Decoding: 16ms/ 1M integers

See benchmarks for details.

### Fibonacci encoding/decoding
Fibonacci encoding is up to speed with other rust implementations, e.g. `fibonnaci_codec` crate (which I took some code from):
- Fibonacci encoding: 
    - this crate: 75ms/ 1M integers 
    - fibonnaci_codec: 88ms / 1M integers
Regular fibonacci decoding (iterator based) is up to speed with the `fibonnaci_codec` crate. 
The FastFibonacci decoding functions are ~2x faster, but have some constant overhead  (i.e. only pays of when decoding *many* integers):
- Fibonacci decoding: 
    - regular decoding: 92ms/ 1M integers
    - fibonnaci_codec: 108ms / 1M integers
    - fast decoding (u8 segments): 40ms / 1M integers
    - fast decoding (u16 segments): 30ms / 1M integers
    - fast decoding (using an iterator): 54ms / 1M integers

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
