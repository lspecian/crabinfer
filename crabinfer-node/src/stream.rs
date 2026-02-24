//! TokenStream — AsyncIterator for streaming token generation.
//!
//! JS usage:
//! ```js
//! const stream = provider.stream({ messages: [...], maxTokens: 200 })
//! for await (const token of stream) {
//!   process.stdout.write(token.text)
//! }
//! ```
//!
//! The `Symbol.asyncIterator` protocol is patched in `index.js` to call
//! `stream.next()` which returns `{ value: TokenOutput, done: boolean }`.

use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::{AsyncTask, Env, Task};
use napi_derive::napi;

use crate::error::to_napi_error;
use crate::types::TokenOutput;

/// The raw core iterator type. Wrapped in a newtype to avoid napi-derive
/// trying to interpret the Result error type.
pub(crate) type CoreTokenIter = Box<
    dyn Iterator<
            Item = std::result::Result<crabinfer_core::TokenOutput, crabinfer_core::CrabInferError>,
        > + Send,
>;

/// Task that pulls the next token from a blocking Rust iterator on the libuv thread pool.
pub struct NextTokenTask {
    iter: Arc<Mutex<Option<CoreTokenIter>>>,
}

#[napi]
impl Task for NextTokenTask {
    type Output = Option<crabinfer_core::TokenOutput>;
    type JsValue = Option<TokenOutput>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        let mut guard = self.iter.lock().map_err(|_| {
            napi::Error::from_reason("Failed to acquire stream lock")
        })?;
        let iter = match guard.as_mut() {
            Some(it) => it,
            None => return Ok(None),
        };
        match iter.next() {
            Some(Ok(token)) => {
                if token.is_end_of_sequence {
                    *guard = None;
                }
                Ok(Some(token))
            }
            Some(Err(e)) => {
                *guard = None;
                Err(to_napi_error(e))
            }
            None => {
                *guard = None;
                Ok(None)
            }
        }
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.map(Into::into))
    }
}

/// Streaming token iterator. Call `next()` to get the next token as a Promise.
/// Implements the async iterator protocol via `index.js` patching.
#[napi]
pub struct TokenStream {
    iter: Arc<Mutex<Option<CoreTokenIter>>>,
}

impl TokenStream {
    pub fn new(iter: CoreTokenIter) -> Self {
        Self {
            iter: Arc::new(Mutex::new(Some(iter))),
        }
    }
}

#[napi]
impl TokenStream {
    /// Get the next token. Returns `Promise<TokenOutput | null>`.
    /// Returns `null` when the stream is exhausted.
    #[napi(ts_return_type = "Promise<TokenOutput | null>")]
    pub fn next(&self) -> AsyncTask<NextTokenTask> {
        AsyncTask::new(NextTokenTask {
            iter: self.iter.clone(),
        })
    }
}
