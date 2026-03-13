//! Disaggregated prefill/decode scheduling for the serving engine.
//!
//! In disaggregated inference, prefill (prompt processing) and decode (token generation)
//! are handled by separate worker pools. This allows:
//! - Independent scaling of compute-bound prefill vs memory-bound decode
//! - Better GPU utilization: prefill workers saturate FLOPs, decode workers saturate bandwidth
//! - Reduced interference between long prefills and latency-sensitive decode steps
//!
//! Architecture:
//! ```text
//! New sequences ──► DisaggregatedScheduler ──► PrefillWorker pool
//!                                                    │
//!                                              KVTransferHandle
//!                                                    │
//!                                              DecodeWorker pool ──► completed tokens
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use super::block::BlockId;
use super::sequence::SeqId;

// ─── Configuration ────────────────────────────────────────────────────────

/// Method for transferring KV cache data between prefill and decode workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVTransferMethod {
    /// Direct memory copy (CPU memcpy or same-device GPU copy).
    Direct,
    /// RDMA-based transfer (stub for future NIC-level zero-copy).
    #[cfg(feature = "cuda")]
    Rdma,
    /// Stage through host memory (GPU→CPU→GPU for cross-device transfers).
    Host,
}

impl Default for KVTransferMethod {
    fn default() -> Self {
        KVTransferMethod::Direct
    }
}

/// Configuration for disaggregated prefill/decode.
#[derive(Debug, Clone)]
pub struct DisaggregatedConfig {
    /// Number of prefill workers.
    pub num_prefill_workers: usize,
    /// Number of decode workers.
    pub num_decode_workers: usize,
    /// KV cache transfer method between prefill and decode pools.
    pub kv_transfer_method: KVTransferMethod,
    /// Block size (tokens per block) — must match KVCacheConfig.
    pub block_size: usize,
    /// Bytes per KV element (for transfer time estimation).
    pub kv_element_bytes: usize,
    /// Number of KV heads per layer.
    pub num_kv_heads: usize,
    /// Head dimension size.
    pub head_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
}

impl Default for DisaggregatedConfig {
    fn default() -> Self {
        Self {
            num_prefill_workers: 1,
            num_decode_workers: 1,
            kv_transfer_method: KVTransferMethod::default(),
            block_size: 16,
            kv_element_bytes: 2,
            num_kv_heads: 8,
            head_size: 128,
            num_layers: 32,
        }
    }
}

impl DisaggregatedConfig {
    /// Bytes of KV data per block (both K and V, all layers).
    pub fn bytes_per_block(&self) -> usize {
        2 * self.num_kv_heads * self.head_size * self.block_size * self.num_layers * self.kv_element_bytes
    }
}

// ─── Worker Role ──────────────────────────────────────────────────────────

/// Role of a worker in the disaggregated system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerRole {
    /// Handles prompt processing (prefill phase).
    Prefill,
    /// Handles autoregressive token generation (decode phase).
    Decode,
}

/// Unique identifier for a worker.
pub type WorkerId = usize;

// ─── KV Transfer ──────────────────────────────────────────────────────────

/// Metadata about KV cache blocks produced by a prefill worker.
#[derive(Debug, Clone)]
pub struct KVBlockMetadata {
    /// Physical block IDs on the source (prefill) worker.
    pub block_ids: Vec<BlockId>,
    /// Sequence ID these blocks belong to.
    pub seq_id: SeqId,
    /// Number of prompt tokens that were processed (prefilled).
    pub num_prefilled_tokens: usize,
    /// Block size used.
    pub block_size: usize,
}

/// Request to transfer KV cache data from a prefill worker to a decode worker.
#[derive(Debug, Clone)]
pub struct KVTransferRequest {
    /// Source worker (prefill).
    pub source_worker: WorkerId,
    /// Destination worker (decode).
    pub dest_worker: WorkerId,
    /// KV block metadata from the prefill phase.
    pub metadata: KVBlockMetadata,
}

/// Result of a KV cache transfer.
#[derive(Debug, Clone)]
pub struct KVTransferResult {
    /// Sequence ID that was transferred.
    pub seq_id: SeqId,
    /// Block IDs on the destination (decode) worker.
    /// In Direct mode these are the same IDs; in Host mode they may be remapped.
    pub dest_block_ids: Vec<BlockId>,
    /// Number of tokens covered by the transferred blocks.
    pub num_tokens: usize,
    /// Estimated transfer duration.
    pub transfer_duration: Duration,
    /// Whether the transfer succeeded.
    pub success: bool,
}

/// Handles KV cache transfers between prefill and decode worker pools.
pub struct KVTransferHandle {
    method: KVTransferMethod,
    /// Bandwidth in bytes/second for transfer time estimation.
    bandwidth_bytes_per_sec: f64,
    /// Bytes per block (computed from config).
    bytes_per_block: usize,
    /// Number of completed transfers.
    completed_transfers: u64,
    /// Total bytes transferred.
    total_bytes_transferred: u64,
}

impl KVTransferHandle {
    /// Create a new KV transfer handle.
    pub fn new(config: &DisaggregatedConfig) -> Self {
        let bandwidth = match config.kv_transfer_method {
            KVTransferMethod::Direct => 100.0e9, // ~100 GB/s (CPU memcpy or PCIe)
            #[cfg(feature = "cuda")]
            KVTransferMethod::Rdma => 200.0e9, // ~200 GB/s (NVLink/IB)
            KVTransferMethod::Host => 25.0e9,  // ~25 GB/s (PCIe staging)
        };

        Self {
            method: config.kv_transfer_method,
            bandwidth_bytes_per_sec: bandwidth,
            bytes_per_block: config.bytes_per_block(),
            completed_transfers: 0,
            total_bytes_transferred: 0,
        }
    }

    /// Estimate the transfer time for a given number of blocks.
    pub fn estimate_transfer_time(&self, num_blocks: usize) -> Duration {
        let total_bytes = num_blocks as f64 * self.bytes_per_block as f64;
        let seconds = total_bytes / self.bandwidth_bytes_per_sec;
        Duration::from_secs_f64(seconds)
    }

    /// Execute a KV cache transfer.
    ///
    /// On CPU this is a logical transfer (metadata copy). On CUDA with the
    /// `cuda` feature, this would invoke device-to-device copies.
    pub fn transfer(&mut self, request: &KVTransferRequest) -> KVTransferResult {
        let num_blocks = request.metadata.block_ids.len();
        let transfer_duration = self.estimate_transfer_time(num_blocks);
        let total_bytes = num_blocks * self.bytes_per_block;

        // On CPU, "transfer" is a no-op since both workers can access the same
        // memory. On CUDA, this would do cudaMemcpyAsync between devices.
        #[cfg(feature = "cuda")]
        {
            // Stub: would call cudaMemcpyPeerAsync here for cross-device transfers.
            let _ = self.method;
        }

        // In Direct mode, dest blocks are the same as source blocks (shared memory).
        // In Host mode, we would remap to destination pool block IDs.
        let dest_block_ids = match self.method {
            KVTransferMethod::Direct => request.metadata.block_ids.clone(),
            #[cfg(feature = "cuda")]
            KVTransferMethod::Rdma => request.metadata.block_ids.clone(),
            KVTransferMethod::Host => {
                // In a real implementation, we'd allocate new blocks on the dest
                // and copy data through host staging. For now, pass through.
                request.metadata.block_ids.clone()
            }
        };

        self.completed_transfers += 1;
        self.total_bytes_transferred += total_bytes as u64;

        KVTransferResult {
            seq_id: request.metadata.seq_id,
            dest_block_ids,
            num_tokens: request.metadata.num_prefilled_tokens,
            transfer_duration,
            success: true,
        }
    }

    /// Number of completed transfers.
    pub fn completed_transfers(&self) -> u64 {
        self.completed_transfers
    }

    /// Total bytes transferred.
    pub fn total_bytes_transferred(&self) -> u64 {
        self.total_bytes_transferred
    }

    /// Transfer method in use.
    pub fn method(&self) -> KVTransferMethod {
        self.method
    }
}

// ─── Prefill Worker ───────────────────────────────────────────────────────

/// State of a sequence within a prefill worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillState {
    /// Sequence is queued, waiting for prefill to start.
    Queued,
    /// Prefill is in progress (chunked prefill may take multiple steps).
    InProgress,
    /// Prefill is complete; KV data is ready for transfer.
    Complete,
}

/// A prefill worker processes prompt tokens for new sequences.
pub struct PrefillWorker {
    /// Worker ID.
    pub id: WorkerId,
    /// Sequences assigned to this worker with their state.
    sequences: HashMap<SeqId, PrefillSequenceState>,
    /// Queue of sequences waiting for prefill.
    queue: VecDeque<SeqId>,
    /// Maximum concurrent prefill sequences.
    max_concurrent: usize,
    /// Number of sequences currently being prefilled.
    active_count: usize,
    /// Total sequences processed by this worker.
    total_processed: u64,
}

/// Per-sequence state tracked by a prefill worker.
#[derive(Debug, Clone)]
pub struct PrefillSequenceState {
    pub seq_id: SeqId,
    pub state: PrefillState,
    /// Number of prompt tokens.
    pub prompt_len: usize,
    /// Block IDs allocated for this sequence's KV cache.
    pub block_ids: Vec<BlockId>,
    /// Number of tokens prefilled so far (for chunked prefill).
    pub tokens_prefilled: usize,
}

impl PrefillWorker {
    /// Create a new prefill worker.
    pub fn new(id: WorkerId, max_concurrent: usize) -> Self {
        Self {
            id,
            sequences: HashMap::new(),
            queue: VecDeque::new(),
            max_concurrent,
            active_count: 0,
            total_processed: 0,
        }
    }

    /// Submit a new sequence for prefill.
    pub fn submit(&mut self, seq_id: SeqId, prompt_len: usize) {
        let state = PrefillSequenceState {
            seq_id,
            state: PrefillState::Queued,
            prompt_len,
            block_ids: Vec::new(),
            tokens_prefilled: 0,
        };
        self.sequences.insert(seq_id, state);
        self.queue.push_back(seq_id);
    }

    /// Start prefill for queued sequences up to the concurrency limit.
    /// Returns the sequence IDs that were started.
    pub fn start_prefill(&mut self) -> Vec<SeqId> {
        let mut started = Vec::new();
        while self.active_count < self.max_concurrent {
            if let Some(seq_id) = self.queue.pop_front() {
                if let Some(state) = self.sequences.get_mut(&seq_id) {
                    state.state = PrefillState::InProgress;
                    self.active_count += 1;
                    started.push(seq_id);
                }
            } else {
                break;
            }
        }
        started
    }

    /// Mark a sequence's prefill as complete with the given KV block IDs.
    pub fn complete_prefill(&mut self, seq_id: SeqId, block_ids: Vec<BlockId>) -> Option<KVBlockMetadata> {
        if let Some(state) = self.sequences.get_mut(&seq_id) {
            state.state = PrefillState::Complete;
            state.block_ids = block_ids.clone();
            state.tokens_prefilled = state.prompt_len;
            self.active_count = self.active_count.saturating_sub(1);
            self.total_processed += 1;

            let block_size = if block_ids.is_empty() { 16 } else { 16 }; // TODO: from config
            Some(KVBlockMetadata {
                block_ids,
                seq_id,
                num_prefilled_tokens: state.prompt_len,
                block_size,
            })
        } else {
            None
        }
    }

    /// Remove a completed sequence from this worker.
    pub fn remove(&mut self, seq_id: SeqId) -> Option<PrefillSequenceState> {
        self.sequences.remove(&seq_id)
    }

    /// Number of sequences currently in this worker (all states).
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Number of sequences actively being prefilled.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Number of sequences queued for prefill.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Total load metric (active + queued).
    pub fn load(&self) -> usize {
        self.active_count + self.queue.len()
    }

    /// Total sequences processed.
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    /// Get the state of a specific sequence.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&PrefillSequenceState> {
        self.sequences.get(&seq_id)
    }

    /// Get all completed sequences.
    pub fn completed_sequences(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| s.state == PrefillState::Complete)
            .map(|s| s.seq_id)
            .collect()
    }
}

// ─── Decode Worker ────────────────────────────────────────────────────────

/// State of a sequence within a decode worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeState {
    /// KV cache is being received (transfer in progress).
    ReceivingKV,
    /// Actively generating tokens.
    Generating,
    /// Generation complete.
    Complete,
}

/// A decode worker handles autoregressive token generation.
pub struct DecodeWorker {
    /// Worker ID.
    pub id: WorkerId,
    /// Sequences assigned to this worker with their state.
    sequences: HashMap<SeqId, DecodeSequenceState>,
    /// Maximum concurrent decode sequences (continuous batching limit).
    max_batch_size: usize,
    /// Total tokens generated by this worker.
    total_tokens_generated: u64,
}

/// Per-sequence state tracked by a decode worker.
#[derive(Debug, Clone)]
pub struct DecodeSequenceState {
    pub seq_id: SeqId,
    pub state: DecodeState,
    /// Block IDs holding the KV cache (received from prefill).
    pub block_ids: Vec<BlockId>,
    /// Number of tokens computed (prompt tokens from prefill + generated).
    pub num_computed_tokens: usize,
    /// Number of tokens generated so far.
    pub num_generated_tokens: usize,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
}

impl DecodeWorker {
    /// Create a new decode worker.
    pub fn new(id: WorkerId, max_batch_size: usize) -> Self {
        Self {
            id,
            sequences: HashMap::new(),
            max_batch_size,
            total_tokens_generated: 0,
        }
    }

    /// Receive a sequence with pre-filled KV cache from a prefill worker.
    pub fn receive_sequence(
        &mut self,
        seq_id: SeqId,
        block_ids: Vec<BlockId>,
        num_prefilled_tokens: usize,
        max_tokens: usize,
    ) -> bool {
        if self.sequences.len() >= self.max_batch_size {
            return false;
        }

        let state = DecodeSequenceState {
            seq_id,
            state: DecodeState::ReceivingKV,
            block_ids,
            num_computed_tokens: num_prefilled_tokens,
            num_generated_tokens: 0,
            max_tokens,
        };
        self.sequences.insert(seq_id, state);
        true
    }

    /// Mark a sequence as ready for generation (KV transfer complete).
    pub fn activate_sequence(&mut self, seq_id: SeqId) -> bool {
        if let Some(state) = self.sequences.get_mut(&seq_id) {
            if state.state == DecodeState::ReceivingKV {
                state.state = DecodeState::Generating;
                return true;
            }
        }
        false
    }

    /// Record a decode step: one token generated for a sequence.
    /// Returns `true` if the sequence should continue, `false` if done.
    pub fn step_sequence(&mut self, seq_id: SeqId) -> bool {
        if let Some(state) = self.sequences.get_mut(&seq_id) {
            if state.state != DecodeState::Generating {
                return false;
            }
            state.num_generated_tokens += 1;
            state.num_computed_tokens += 1;
            self.total_tokens_generated += 1;

            if state.num_generated_tokens >= state.max_tokens {
                state.state = DecodeState::Complete;
                return false;
            }
            true
        } else {
            false
        }
    }

    /// Remove a completed sequence.
    pub fn remove(&mut self, seq_id: SeqId) -> Option<DecodeSequenceState> {
        self.sequences.remove(&seq_id)
    }

    /// Number of sequences currently in this worker.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Number of sequences actively generating tokens.
    pub fn active_generating(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| s.state == DecodeState::Generating)
            .count()
    }

    /// Total load metric (all sequences regardless of state).
    pub fn load(&self) -> usize {
        self.sequences.len()
    }

    /// Total tokens generated by this worker.
    pub fn total_tokens_generated(&self) -> u64 {
        self.total_tokens_generated
    }

    /// Whether this worker can accept more sequences.
    pub fn has_capacity(&self) -> bool {
        self.sequences.len() < self.max_batch_size
    }

    /// Get a sequence's state.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&DecodeSequenceState> {
        self.sequences.get(&seq_id)
    }

    /// Get all completed sequence IDs.
    pub fn completed_sequences(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| s.state == DecodeState::Complete)
            .map(|s| s.seq_id)
            .collect()
    }

    /// Get all actively generating sequence IDs.
    pub fn generating_sequences(&self) -> Vec<SeqId> {
        self.sequences
            .values()
            .filter(|s| s.state == DecodeState::Generating)
            .map(|s| s.seq_id)
            .collect()
    }
}

// ─── Disaggregated Scheduler ──────────────────────────────────────────────

/// Batch assignment for a single scheduling step.
#[derive(Debug)]
pub struct DisaggregatedScheduleResult {
    /// Sequences assigned to prefill workers: (seq_id, worker_id).
    pub prefill_assignments: Vec<(SeqId, WorkerId)>,
    /// Sequences assigned to decode workers: (seq_id, worker_id).
    pub decode_assignments: Vec<(SeqId, WorkerId)>,
    /// Sequences whose KV cache needs to be transferred.
    pub pending_transfers: Vec<KVTransferRequest>,
}

/// Routes sequences to the appropriate worker pool based on their lifecycle stage.
pub struct DisaggregatedScheduler {
    config: DisaggregatedConfig,
    /// Load tracking: worker_id → current load for prefill workers.
    prefill_loads: Vec<usize>,
    /// Load tracking: worker_id → current load for decode workers.
    decode_loads: Vec<usize>,
}

impl DisaggregatedScheduler {
    /// Create a new disaggregated scheduler.
    pub fn new(config: DisaggregatedConfig) -> Self {
        let prefill_loads = vec![0; config.num_prefill_workers];
        let decode_loads = vec![0; config.num_decode_workers];
        Self {
            config,
            prefill_loads,
            decode_loads,
        }
    }

    /// Select the least-loaded prefill worker.
    pub fn select_prefill_worker(&self) -> WorkerId {
        self.prefill_loads
            .iter()
            .enumerate()
            .min_by_key(|&(_, load)| load)
            .map(|(id, _)| id)
            .unwrap_or(0)
    }

    /// Select the least-loaded decode worker.
    pub fn select_decode_worker(&self) -> WorkerId {
        self.decode_loads
            .iter()
            .enumerate()
            .min_by_key(|&(_, load)| load)
            .map(|(id, _)| id)
            .unwrap_or(0)
    }

    /// Update the load for a prefill worker.
    pub fn update_prefill_load(&mut self, worker_id: WorkerId, load: usize) {
        if worker_id < self.prefill_loads.len() {
            self.prefill_loads[worker_id] = load;
        }
    }

    /// Update the load for a decode worker.
    pub fn update_decode_load(&mut self, worker_id: WorkerId, load: usize) {
        if worker_id < self.decode_loads.len() {
            self.decode_loads[worker_id] = load;
        }
    }

    /// Schedule new and post-prefill sequences to appropriate workers.
    ///
    /// - `new_seqs`: newly arrived sequences needing prefill (seq_id, prompt_len).
    /// - `completed_prefills`: sequences that finished prefill, needing decode assignment.
    ///   Each entry is (seq_id, source_prefill_worker, kv_metadata).
    pub fn schedule(
        &mut self,
        new_seqs: &[(SeqId, usize)],
        completed_prefills: &[(SeqId, WorkerId, KVBlockMetadata)],
    ) -> DisaggregatedScheduleResult {
        let mut prefill_assignments = Vec::new();
        let mut decode_assignments = Vec::new();
        let mut pending_transfers = Vec::new();

        // Route new sequences to the least-loaded prefill worker.
        for &(seq_id, _prompt_len) in new_seqs {
            let worker_id = self.select_prefill_worker();
            prefill_assignments.push((seq_id, worker_id));
            self.prefill_loads[worker_id] += 1;
        }

        // Route completed prefills to the least-loaded decode worker.
        for (seq_id, source_worker, metadata) in completed_prefills {
            let dest_worker = self.select_decode_worker();
            decode_assignments.push((*seq_id, dest_worker));
            self.decode_loads[dest_worker] += 1;

            pending_transfers.push(KVTransferRequest {
                source_worker: *source_worker,
                dest_worker,
                metadata: metadata.clone(),
            });
        }

        DisaggregatedScheduleResult {
            prefill_assignments,
            decode_assignments,
            pending_transfers,
        }
    }

    /// Get the current prefill worker loads.
    pub fn prefill_loads(&self) -> &[usize] {
        &self.prefill_loads
    }

    /// Get the current decode worker loads.
    pub fn decode_loads(&self) -> &[usize] {
        &self.decode_loads
    }

    /// Number of prefill workers.
    pub fn num_prefill_workers(&self) -> usize {
        self.config.num_prefill_workers
    }

    /// Number of decode workers.
    pub fn num_decode_workers(&self) -> usize {
        self.config.num_decode_workers
    }
}

// ─── Disaggregated Coordinator ────────────────────────────────────────────

/// Lifecycle stage of a sequence in the disaggregated pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStage {
    /// Assigned to a prefill worker, awaiting prefill.
    Prefilling,
    /// Prefill complete, KV cache being transferred.
    Transferring,
    /// Assigned to a decode worker, generating tokens.
    Decoding,
    /// Generation complete.
    Complete,
}

/// Tracking info for a sequence in the coordinator.
#[derive(Debug, Clone)]
pub struct CoordinatorSequence {
    pub seq_id: SeqId,
    pub stage: SequenceStage,
    pub prompt_len: usize,
    pub max_tokens: usize,
    /// Which prefill worker is handling/handled this sequence.
    pub prefill_worker: Option<WorkerId>,
    /// Which decode worker is handling this sequence.
    pub decode_worker: Option<WorkerId>,
}

/// Top-level orchestrator for disaggregated prefill/decode.
///
/// Manages the full lifecycle: submit → prefill → transfer → decode → complete.
pub struct DisaggregatedCoordinator {
    config: DisaggregatedConfig,
    scheduler: DisaggregatedScheduler,
    transfer_handle: KVTransferHandle,
    prefill_workers: Vec<PrefillWorker>,
    decode_workers: Vec<DecodeWorker>,
    /// All sequences tracked by the coordinator.
    sequences: HashMap<SeqId, CoordinatorSequence>,
    /// Completed sequence IDs, drained by the caller.
    completed: VecDeque<SeqId>,
    /// Next sequence ID to assign.
    next_seq_id: SeqId,
}

impl DisaggregatedCoordinator {
    /// Create a new coordinator with the given config.
    pub fn new(config: DisaggregatedConfig) -> Self {
        let scheduler = DisaggregatedScheduler::new(config.clone());
        let transfer_handle = KVTransferHandle::new(&config);

        let prefill_workers = (0..config.num_prefill_workers)
            .map(|id| PrefillWorker::new(id, 4)) // max 4 concurrent prefills per worker
            .collect();
        let decode_workers = (0..config.num_decode_workers)
            .map(|id| DecodeWorker::new(id, 64)) // max 64 concurrent decodes per worker
            .collect();

        Self {
            config,
            scheduler,
            transfer_handle,
            prefill_workers,
            decode_workers,
            sequences: HashMap::new(),
            completed: VecDeque::new(),
            next_seq_id: 1,
        }
    }

    /// Submit a new sequence for processing.
    /// Returns the assigned sequence ID.
    pub fn submit(&mut self, prompt_len: usize, max_tokens: usize) -> SeqId {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let coord_seq = CoordinatorSequence {
            seq_id,
            stage: SequenceStage::Prefilling,
            prompt_len,
            max_tokens,
            prefill_worker: None,
            decode_worker: None,
        };
        self.sequences.insert(seq_id, coord_seq);

        // Schedule to a prefill worker.
        let result = self.scheduler.schedule(&[(seq_id, prompt_len)], &[]);
        for (sid, worker_id) in &result.prefill_assignments {
            if let Some(cs) = self.sequences.get_mut(sid) {
                cs.prefill_worker = Some(*worker_id);
            }
            if let Some(worker) = self.prefill_workers.get_mut(*worker_id) {
                worker.submit(*sid, prompt_len);
            }
        }

        seq_id
    }

    /// Run a single step of the pipeline:
    /// 1. Start queued prefills on workers.
    /// 2. (Caller simulates prefill completion via `complete_prefill`.)
    /// 3. Transfer completed prefills to decode workers.
    /// 4. Step decode workers.
    /// 5. Drain completed sequences.
    pub fn step(&mut self) {
        // 1. Start queued prefills on each worker.
        for worker in &mut self.prefill_workers {
            worker.start_prefill();
        }

        // 2. Collect completed prefills and transfer to decode.
        let mut completed_prefills: Vec<(SeqId, WorkerId, KVBlockMetadata)> = Vec::new();
        for worker in &self.prefill_workers {
            for seq_id in worker.completed_sequences() {
                if let Some(seq_state) = worker.get_sequence(seq_id) {
                    let metadata = KVBlockMetadata {
                        block_ids: seq_state.block_ids.clone(),
                        seq_id,
                        num_prefilled_tokens: seq_state.prompt_len,
                        block_size: self.config.block_size,
                    };
                    completed_prefills.push((seq_id, worker.id, metadata));
                }
            }
        }

        // Schedule completed prefills to decode workers.
        if !completed_prefills.is_empty() {
            let result = self.scheduler.schedule(&[], &completed_prefills);

            // Execute transfers.
            for request in &result.pending_transfers {
                let transfer_result = self.transfer_handle.transfer(request);
                if transfer_result.success {
                    let seq_id = transfer_result.seq_id;

                    // Update coordinator stage.
                    if let Some(cs) = self.sequences.get_mut(&seq_id) {
                        cs.stage = SequenceStage::Decoding;
                        cs.decode_worker = Some(request.dest_worker);
                    }

                    // Receive on decode worker.
                    let max_tokens = self.sequences.get(&seq_id)
                        .map(|s| s.max_tokens)
                        .unwrap_or(128);
                    if let Some(dw) = self.decode_workers.get_mut(request.dest_worker) {
                        dw.receive_sequence(
                            seq_id,
                            transfer_result.dest_block_ids,
                            transfer_result.num_tokens,
                            max_tokens,
                        );
                        dw.activate_sequence(seq_id);
                    }

                    // Remove from prefill worker.
                    if let Some(pw) = self.prefill_workers.get_mut(request.source_worker) {
                        pw.remove(seq_id);
                    }
                }
            }
        }

        // 3. Step all decode workers: generate one token per active sequence.
        for worker in &mut self.decode_workers {
            let generating: Vec<SeqId> = worker.generating_sequences();
            for seq_id in generating {
                worker.step_sequence(seq_id);
            }

            // Drain completed decode sequences.
            let done: Vec<SeqId> = worker.completed_sequences();
            for seq_id in done {
                worker.remove(seq_id);
                if let Some(cs) = self.sequences.get_mut(&seq_id) {
                    cs.stage = SequenceStage::Complete;
                }
                self.completed.push_back(seq_id);

                // Update decode load.
                self.scheduler.update_decode_load(worker.id, worker.load());
            }
        }

        // 4. Update load metrics.
        for worker in &self.prefill_workers {
            self.scheduler.update_prefill_load(worker.id, worker.load());
        }
        for worker in &self.decode_workers {
            self.scheduler.update_decode_load(worker.id, worker.load());
        }
    }

    /// Simulate completing prefill for a sequence on its assigned worker.
    /// In a real system, the model forward pass signals this.
    pub fn complete_prefill(&mut self, seq_id: SeqId, block_ids: Vec<BlockId>) {
        if let Some(cs) = self.sequences.get_mut(&seq_id) {
            cs.stage = SequenceStage::Transferring;
            if let Some(worker_id) = cs.prefill_worker {
                if let Some(worker) = self.prefill_workers.get_mut(worker_id) {
                    worker.complete_prefill(seq_id, block_ids);
                }
            }
        }
    }

    /// Drain completed sequences.
    pub fn drain_completed(&mut self) -> Vec<SeqId> {
        self.completed.drain(..).collect()
    }

    /// Get a sequence's current stage.
    pub fn get_stage(&self, seq_id: SeqId) -> Option<SequenceStage> {
        self.sequences.get(&seq_id).map(|s| s.stage)
    }

    /// Total number of tracked sequences (all stages).
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Number of sequences in a given stage.
    pub fn num_in_stage(&self, stage: SequenceStage) -> usize {
        self.sequences.values().filter(|s| s.stage == stage).count()
    }

    /// Access the transfer handle (for stats).
    pub fn transfer_handle(&self) -> &KVTransferHandle {
        &self.transfer_handle
    }

    /// Access the scheduler (for load info).
    pub fn scheduler(&self) -> &DisaggregatedScheduler {
        &self.scheduler
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DisaggregatedConfig {
        DisaggregatedConfig {
            num_prefill_workers: 2,
            num_decode_workers: 2,
            kv_transfer_method: KVTransferMethod::Direct,
            block_size: 16,
            kv_element_bytes: 2,
            num_kv_heads: 8,
            head_size: 128,
            num_layers: 32,
        }
    }

    // ── Worker Role Tests ──────────────────────────────────────────────

    #[test]
    fn test_worker_role_equality() {
        assert_eq!(WorkerRole::Prefill, WorkerRole::Prefill);
        assert_eq!(WorkerRole::Decode, WorkerRole::Decode);
        assert_ne!(WorkerRole::Prefill, WorkerRole::Decode);
    }

    #[test]
    fn test_prefill_worker_only_prefills() {
        let mut worker = PrefillWorker::new(0, 4);
        worker.submit(1, 100);
        worker.submit(2, 200);

        assert_eq!(worker.num_sequences(), 2);
        assert_eq!(worker.queue_len(), 2);
        assert_eq!(worker.active_count(), 0);

        // Start prefill — should move from queue to active.
        let started = worker.start_prefill();
        assert_eq!(started.len(), 2);
        assert_eq!(worker.active_count(), 2);
        assert_eq!(worker.queue_len(), 0);

        // Complete one prefill.
        let meta = worker.complete_prefill(1, vec![10, 11, 12]).unwrap();
        assert_eq!(meta.seq_id, 1);
        assert_eq!(meta.num_prefilled_tokens, 100);
        assert_eq!(meta.block_ids, vec![10, 11, 12]);
        assert_eq!(worker.active_count(), 1);
    }

    #[test]
    fn test_decode_worker_only_decodes() {
        let mut worker = DecodeWorker::new(0, 64);

        // Receive a pre-filled sequence.
        assert!(worker.receive_sequence(1, vec![10, 11], 100, 5));
        assert_eq!(worker.num_sequences(), 1);
        assert_eq!(worker.active_generating(), 0);

        // Activate it.
        assert!(worker.activate_sequence(1));
        assert_eq!(worker.active_generating(), 1);

        // Decode steps.
        assert!(worker.step_sequence(1)); // token 1
        assert!(worker.step_sequence(1)); // token 2
        assert!(worker.step_sequence(1)); // token 3
        assert!(worker.step_sequence(1)); // token 4
        assert!(!worker.step_sequence(1)); // token 5 — hits max, complete

        let state = worker.get_sequence(1).unwrap();
        assert_eq!(state.state, DecodeState::Complete);
        assert_eq!(state.num_generated_tokens, 5);
    }

    #[test]
    fn test_prefill_worker_concurrency_limit() {
        let mut worker = PrefillWorker::new(0, 2);
        worker.submit(1, 100);
        worker.submit(2, 100);
        worker.submit(3, 100);

        let started = worker.start_prefill();
        assert_eq!(started.len(), 2);
        assert_eq!(worker.queue_len(), 1);
        assert_eq!(worker.active_count(), 2);

        // Can't start more until one completes.
        let started2 = worker.start_prefill();
        assert!(started2.is_empty());

        // Complete one → can start the queued one.
        worker.complete_prefill(1, vec![]);
        let started3 = worker.start_prefill();
        assert_eq!(started3.len(), 1);
        assert_eq!(started3[0], 3);
    }

    #[test]
    fn test_decode_worker_capacity_limit() {
        let mut worker = DecodeWorker::new(0, 2);
        assert!(worker.receive_sequence(1, vec![], 10, 5));
        assert!(worker.receive_sequence(2, vec![], 10, 5));
        assert!(!worker.receive_sequence(3, vec![], 10, 5)); // full
        assert!(!worker.has_capacity());
    }

    // ── KV Transfer Tests ─────────────────────────────────────────────

    #[test]
    fn test_kv_transfer_preserves_metadata() {
        let config = test_config();
        let mut handle = KVTransferHandle::new(&config);

        let metadata = KVBlockMetadata {
            block_ids: vec![5, 6, 7],
            seq_id: 42,
            num_prefilled_tokens: 48,
            block_size: 16,
        };

        let request = KVTransferRequest {
            source_worker: 0,
            dest_worker: 1,
            metadata,
        };

        let result = handle.transfer(&request);
        assert!(result.success);
        assert_eq!(result.seq_id, 42);
        assert_eq!(result.dest_block_ids, vec![5, 6, 7]);
        assert_eq!(result.num_tokens, 48);
    }

    #[test]
    fn test_kv_transfer_time_estimation() {
        let config = test_config();
        let handle = KVTransferHandle::new(&config);

        let t1 = handle.estimate_transfer_time(1);
        let t10 = handle.estimate_transfer_time(10);

        // 10 blocks should take ~10x longer than 1 block.
        let ratio = t10.as_secs_f64() / t1.as_secs_f64();
        assert!((ratio - 10.0).abs() < 0.01, "ratio should be ~10, got {ratio}");

        // Should be non-zero.
        assert!(t1.as_nanos() > 0);
    }

    #[test]
    fn test_kv_transfer_time_scales_with_block_size() {
        let mut config1 = test_config();
        config1.block_size = 16;
        let handle1 = KVTransferHandle::new(&config1);
        let t1 = handle1.estimate_transfer_time(1);

        let mut config2 = test_config();
        config2.block_size = 32;
        let handle2 = KVTransferHandle::new(&config2);
        let t2 = handle2.estimate_transfer_time(1);

        // Doubling block_size should ~double transfer time per block.
        let ratio = t2.as_secs_f64() / t1.as_secs_f64();
        assert!((ratio - 2.0).abs() < 0.01, "ratio should be ~2, got {ratio}");
    }

    #[test]
    fn test_kv_transfer_stats_tracking() {
        let config = test_config();
        let mut handle = KVTransferHandle::new(&config);
        assert_eq!(handle.completed_transfers(), 0);
        assert_eq!(handle.total_bytes_transferred(), 0);

        let request = KVTransferRequest {
            source_worker: 0,
            dest_worker: 1,
            metadata: KVBlockMetadata {
                block_ids: vec![0, 1],
                seq_id: 1,
                num_prefilled_tokens: 32,
                block_size: 16,
            },
        };

        handle.transfer(&request);
        assert_eq!(handle.completed_transfers(), 1);
        assert_eq!(handle.total_bytes_transferred(), 2 * config.bytes_per_block() as u64);

        handle.transfer(&request);
        assert_eq!(handle.completed_transfers(), 2);
    }

    #[test]
    fn test_kv_transfer_host_method() {
        let mut config = test_config();
        config.kv_transfer_method = KVTransferMethod::Host;
        let mut handle = KVTransferHandle::new(&config);
        assert_eq!(handle.method(), KVTransferMethod::Host);

        let request = KVTransferRequest {
            source_worker: 0,
            dest_worker: 1,
            metadata: KVBlockMetadata {
                block_ids: vec![0],
                seq_id: 1,
                num_prefilled_tokens: 16,
                block_size: 16,
            },
        };

        let result = handle.transfer(&request);
        assert!(result.success);
        // Host method has lower bandwidth → longer transfer time.
        let direct_config = test_config();
        let direct_handle = KVTransferHandle::new(&direct_config);
        let direct_time = direct_handle.estimate_transfer_time(1);
        assert!(result.transfer_duration > direct_time);
    }

    // ── Scheduler Tests ───────────────────────────────────────────────

    #[test]
    fn test_scheduler_routes_new_to_prefill() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        let result = scheduler.schedule(&[(1, 100), (2, 200)], &[]);
        assert_eq!(result.prefill_assignments.len(), 2);
        assert!(result.decode_assignments.is_empty());
        assert!(result.pending_transfers.is_empty());
    }

    #[test]
    fn test_scheduler_routes_completed_to_decode() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        let metadata = KVBlockMetadata {
            block_ids: vec![0, 1],
            seq_id: 1,
            num_prefilled_tokens: 32,
            block_size: 16,
        };

        let result = scheduler.schedule(&[], &[(1, 0, metadata)]);
        assert!(result.prefill_assignments.is_empty());
        assert_eq!(result.decode_assignments.len(), 1);
        assert_eq!(result.pending_transfers.len(), 1);
    }

    #[test]
    fn test_scheduler_least_loaded_prefill() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        // Worker 0 has load 5, worker 1 has load 2.
        scheduler.update_prefill_load(0, 5);
        scheduler.update_prefill_load(1, 2);

        let worker = scheduler.select_prefill_worker();
        assert_eq!(worker, 1);
    }

    #[test]
    fn test_scheduler_least_loaded_decode() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        scheduler.update_decode_load(0, 10);
        scheduler.update_decode_load(1, 3);

        let worker = scheduler.select_decode_worker();
        assert_eq!(worker, 1);
    }

    #[test]
    fn test_scheduler_even_distribution() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        // Schedule 4 sequences — should distribute evenly across 2 workers.
        let new_seqs: Vec<(SeqId, usize)> = (1..=4).map(|id| (id, 100)).collect();
        let result = scheduler.schedule(&new_seqs, &[]);

        let mut counts = [0usize; 2];
        for (_, worker_id) in &result.prefill_assignments {
            counts[*worker_id] += 1;
        }
        assert_eq!(counts[0], 2);
        assert_eq!(counts[1], 2);
    }

    #[test]
    fn test_scheduler_mixed_new_and_completed() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        let metadata = KVBlockMetadata {
            block_ids: vec![0],
            seq_id: 1,
            num_prefilled_tokens: 16,
            block_size: 16,
        };

        let result = scheduler.schedule(
            &[(2, 100), (3, 200)],
            &[(1, 0, metadata)],
        );

        assert_eq!(result.prefill_assignments.len(), 2);
        assert_eq!(result.decode_assignments.len(), 1);
        assert_eq!(result.pending_transfers.len(), 1);
    }

    // ── Coordinator / End-to-End Tests ────────────────────────────────

    #[test]
    fn test_coordinator_submit() {
        let config = test_config();
        let mut coord = DisaggregatedCoordinator::new(config);

        let seq_id = coord.submit(100, 10);
        assert_eq!(seq_id, 1);
        assert_eq!(coord.num_sequences(), 1);
        assert_eq!(coord.get_stage(seq_id), Some(SequenceStage::Prefilling));
    }

    #[test]
    fn test_coordinator_end_to_end_lifecycle() {
        let mut config = test_config();
        config.num_prefill_workers = 1;
        config.num_decode_workers = 1;
        let mut coord = DisaggregatedCoordinator::new(config);

        // 1. Submit a sequence.
        let seq_id = coord.submit(32, 3); // 3 max decode tokens
        assert_eq!(coord.get_stage(seq_id), Some(SequenceStage::Prefilling));

        // 2. Step to start prefill.
        coord.step();

        // 3. Simulate prefill completion.
        coord.complete_prefill(seq_id, vec![0, 1]);
        assert_eq!(coord.get_stage(seq_id), Some(SequenceStage::Transferring));

        // 4. Step to transfer and start decode.
        coord.step();
        assert_eq!(coord.get_stage(seq_id), Some(SequenceStage::Decoding));

        // 5. Step 3 times to generate 3 tokens → complete.
        coord.step(); // token 1
        coord.step(); // token 2
        coord.step(); // token 3 → done

        let completed = coord.drain_completed();
        assert!(completed.contains(&seq_id));
        assert_eq!(coord.get_stage(seq_id), Some(SequenceStage::Complete));
    }

    #[test]
    fn test_coordinator_multiple_sequences() {
        let mut config = test_config();
        config.num_prefill_workers = 2;
        config.num_decode_workers = 2;
        let mut coord = DisaggregatedCoordinator::new(config);

        let s1 = coord.submit(32, 2);
        let s2 = coord.submit(64, 2);

        // Start prefills.
        coord.step();

        // Complete both prefills.
        coord.complete_prefill(s1, vec![0, 1]);
        coord.complete_prefill(s2, vec![2, 3, 4, 5]);

        // Transfer + start decode.
        coord.step();

        assert_eq!(coord.get_stage(s1), Some(SequenceStage::Decoding));
        assert_eq!(coord.get_stage(s2), Some(SequenceStage::Decoding));

        // Generate tokens.
        coord.step(); // token 1
        coord.step(); // token 2 → both done

        let completed = coord.drain_completed();
        assert_eq!(completed.len(), 2);
    }

    #[test]
    fn test_coordinator_single_worker_each() {
        let mut config = test_config();
        config.num_prefill_workers = 1;
        config.num_decode_workers = 1;
        let mut coord = DisaggregatedCoordinator::new(config);

        let seq_id = coord.submit(16, 1);
        coord.step();
        coord.complete_prefill(seq_id, vec![0]);
        coord.step(); // transfer + activate
        coord.step(); // generate 1 token → done

        let completed = coord.drain_completed();
        assert_eq!(completed, vec![seq_id]);
    }

    #[test]
    fn test_coordinator_transfer_stats() {
        let config = test_config();
        let mut coord = DisaggregatedCoordinator::new(config);

        let seq_id = coord.submit(32, 1);
        coord.step();
        coord.complete_prefill(seq_id, vec![0, 1]);
        coord.step(); // triggers transfer

        assert_eq!(coord.transfer_handle().completed_transfers(), 1);
        assert!(coord.transfer_handle().total_bytes_transferred() > 0);
    }

    #[test]
    fn test_coordinator_stages_count() {
        let config = test_config();
        let mut coord = DisaggregatedCoordinator::new(config);

        coord.submit(32, 2);
        coord.submit(64, 2);

        assert_eq!(coord.num_in_stage(SequenceStage::Prefilling), 2);
        assert_eq!(coord.num_in_stage(SequenceStage::Decoding), 0);
    }

    #[test]
    fn test_prefill_worker_completed_sequences_list() {
        let mut worker = PrefillWorker::new(0, 4);
        worker.submit(1, 50);
        worker.submit(2, 100);
        worker.start_prefill();

        assert!(worker.completed_sequences().is_empty());

        worker.complete_prefill(1, vec![0, 1]);
        let completed = worker.completed_sequences();
        assert_eq!(completed.len(), 1);
        assert!(completed.contains(&1));
    }

    #[test]
    fn test_decode_worker_generating_sequences_list() {
        let mut worker = DecodeWorker::new(0, 64);
        worker.receive_sequence(1, vec![0], 16, 5);
        worker.receive_sequence(2, vec![1], 16, 5);
        worker.activate_sequence(1);

        let generating = worker.generating_sequences();
        assert_eq!(generating.len(), 1);
        assert!(generating.contains(&1));
    }

    #[test]
    fn test_decode_worker_step_inactive_returns_false() {
        let mut worker = DecodeWorker::new(0, 64);
        worker.receive_sequence(1, vec![0], 16, 5);
        // Not activated yet — step should return false.
        assert!(!worker.step_sequence(1));
    }

    #[test]
    fn test_decode_worker_step_nonexistent_returns_false() {
        let mut worker = DecodeWorker::new(0, 64);
        assert!(!worker.step_sequence(999));
    }

    #[test]
    fn test_config_bytes_per_block() {
        let config = test_config();
        // 2 (K+V) * 8 heads * 128 dim * 16 tokens * 32 layers * 2 bytes
        let expected = 2 * 8 * 128 * 16 * 32 * 2;
        assert_eq!(config.bytes_per_block(), expected);
    }

    #[test]
    fn test_config_default() {
        let config = DisaggregatedConfig::default();
        assert_eq!(config.num_prefill_workers, 1);
        assert_eq!(config.num_decode_workers, 1);
        assert_eq!(config.kv_transfer_method, KVTransferMethod::Direct);
    }

    #[test]
    fn test_prefill_worker_load_metric() {
        let mut worker = PrefillWorker::new(0, 4);
        assert_eq!(worker.load(), 0);

        worker.submit(1, 100);
        assert_eq!(worker.load(), 1); // queued

        worker.start_prefill();
        assert_eq!(worker.load(), 1); // active (moved from queue)

        worker.submit(2, 200);
        assert_eq!(worker.load(), 2); // 1 active + 1 queued
    }

    #[test]
    fn test_decode_worker_load_metric() {
        let mut worker = DecodeWorker::new(0, 64);
        assert_eq!(worker.load(), 0);

        worker.receive_sequence(1, vec![], 10, 5);
        assert_eq!(worker.load(), 1);

        worker.receive_sequence(2, vec![], 10, 5);
        assert_eq!(worker.load(), 2);

        worker.remove(1);
        assert_eq!(worker.load(), 1);
    }

    #[test]
    fn test_prefill_worker_remove() {
        let mut worker = PrefillWorker::new(0, 4);
        worker.submit(1, 100);
        worker.start_prefill();
        worker.complete_prefill(1, vec![0]);

        let removed = worker.remove(1);
        assert!(removed.is_some());
        assert_eq!(worker.num_sequences(), 0);
    }

    #[test]
    fn test_decode_worker_remove() {
        let mut worker = DecodeWorker::new(0, 64);
        worker.receive_sequence(1, vec![0], 16, 5);
        worker.activate_sequence(1);

        let removed = worker.remove(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().state, DecodeState::Generating);
        assert_eq!(worker.num_sequences(), 0);
    }

    #[test]
    fn test_scheduler_load_tracking() {
        let config = test_config();
        let mut scheduler = DisaggregatedScheduler::new(config);

        scheduler.update_prefill_load(0, 3);
        scheduler.update_prefill_load(1, 7);
        assert_eq!(scheduler.prefill_loads(), &[3, 7]);

        scheduler.update_decode_load(0, 10);
        scheduler.update_decode_load(1, 2);
        assert_eq!(scheduler.decode_loads(), &[10, 2]);
    }

    #[test]
    fn test_coordinator_drain_completed_empty() {
        let config = test_config();
        let mut coord = DisaggregatedCoordinator::new(config);
        assert!(coord.drain_completed().is_empty());
    }

    #[test]
    fn test_prefill_worker_total_processed() {
        let mut worker = PrefillWorker::new(0, 4);
        assert_eq!(worker.total_processed(), 0);

        worker.submit(1, 100);
        worker.start_prefill();
        worker.complete_prefill(1, vec![0]);
        assert_eq!(worker.total_processed(), 1);

        worker.submit(2, 200);
        worker.start_prefill();
        worker.complete_prefill(2, vec![1, 2]);
        assert_eq!(worker.total_processed(), 2);
    }

    #[test]
    fn test_decode_worker_total_tokens_generated() {
        let mut worker = DecodeWorker::new(0, 64);
        assert_eq!(worker.total_tokens_generated(), 0);

        worker.receive_sequence(1, vec![], 10, 3);
        worker.activate_sequence(1);
        worker.step_sequence(1);
        worker.step_sequence(1);
        assert_eq!(worker.total_tokens_generated(), 2);
    }
}
