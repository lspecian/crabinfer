//! Backend implementations.
//!
//! Each sub-module provides a concrete [`Backend`](crate::backend::Backend)
//! implementation for a specific ML runtime.

pub mod candle;

pub use candle::CandleBackend;
