//! Smart Router — automatically selects between local, self-hosted, and cloud providers.
//!
//! The Router implements the `Provider` trait, making it a drop-in replacement.
//! Routing decisions are based on memory pressure, model availability, network
//! status, provider tier, latency, and a configurable routing policy.

use crate::memory::quick_check_pressure;
use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider,
};
use crate::{CrabInferError, MemoryPressure, TokenOutput};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Types (UniFFI-exported)
// ---------------------------------------------------------------------------

/// Which infrastructure tier a provider belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ProviderTier {
    /// On-device inference (CrabInferEngine). Lowest latency, highest privacy.
    Local,
    /// Self-hosted server on LAN or private infra (Ollama, vLLM, CrabInfer Server).
    SelfHosted,
    /// Public cloud API (OpenAI, Anthropic, Google).
    Cloud,
}

/// Routing policy controlling how providers are selected.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum RoutingPolicy {
    /// Prefer local inference. Fall back to self-hosted, then cloud.
    LocalFirst,
    /// Prefer cloud inference. Fall back to self-hosted, then local.
    CloudFirst,
    /// Never use cloud or self-hosted providers. Error if local inference fails.
    LocalOnly,
    /// Three-tier cascade: local → self-hosted (by latency) → cloud.
    /// Uses latency-aware selection for self-hosted providers.
    Auto,
    /// Prefer self-hosted servers. Fall back to local, then cloud.
    SelfHostedFirst,
}

/// Why a specific provider was chosen for a request.
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum RoutingReason {
    /// Local model is available and memory pressure is Normal.
    LocalAvailable,
    /// Memory pressure forced fallback from local.
    MemoryPressureFallback,
    /// No local model loaded.
    NoLocalModel,
    /// Cloud preferred by policy.
    CloudPreferred,
    /// Network unavailable, forced local.
    NetworkUnavailable,
    /// Privacy mode: cloud and self-hosted blocked by configuration.
    PrivacyMode,
    /// Primary provider failed, using fallback.
    FallbackAfterError,
    /// Self-hosted provider selected based on availability and latency.
    SelfHostedAvailable,
    /// Self-hosted provider preferred by policy.
    SelfHostedPreferred,
    /// Data sovereignty: cloud blocked, using self-hosted or local.
    DataSovereignty,
}

/// Details about a routing decision.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RoutingDecision {
    /// Name of the provider that was selected.
    pub provider_name: String,
    /// Why this provider was selected.
    pub reason: RoutingReason,
    /// Number of providers that were tried.
    pub providers_tried: u32,
    /// Whether the selected provider is local (kept for backward compat).
    pub is_local: bool,
    /// Which tier the selected provider belongs to.
    pub provider_tier: ProviderTier,
    /// Memory pressure at routing time.
    pub memory_pressure: MemoryPressure,
    /// Whether network was considered available.
    pub network_available: bool,
    /// Health-check RTT in milliseconds to the selected provider (0 if not measured).
    pub latency_ms: u32,
}

/// Router configuration.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RouterConfig {
    /// Routing policy.
    pub policy: RoutingPolicy,
    /// Privacy mode: if true, never route to cloud or self-hosted regardless of policy.
    pub privacy_mode: bool,
    /// Whether to treat Ollama as a local provider (reserved for v2).
    pub ollama_is_local: bool,
    /// Data sovereignty: if true, never route to public cloud. Self-hosted is still allowed.
    pub data_sovereignty: bool,
}

// ---------------------------------------------------------------------------
// Internal provider wrapper with metadata
// ---------------------------------------------------------------------------

/// A provider with its tier classification and last-measured latency.
struct ProviderWithMeta {
    provider: Box<dyn Provider>,
    #[allow(dead_code)]
    tier: ProviderTier,
    /// Last measured health-check latency in milliseconds. u32::MAX = not measured.
    latency_ms: AtomicU32,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Latency refresh interval in milliseconds (60 seconds).
const LATENCY_STALE_MS: u64 = 60_000;

/// Smart router that selects between local, self-hosted, and cloud providers per-request.
pub struct Router {
    /// The local provider (wrapping CrabInferEngine). None if not configured.
    local: Option<Box<dyn Provider>>,
    /// Self-hosted providers (Ollama, vLLM, etc.), in preference order.
    self_hosted: Vec<ProviderWithMeta>,
    /// Cloud providers (OpenAI, Anthropic, Google), in preference order.
    cloud: Vec<ProviderWithMeta>,
    /// Configuration.
    config: RouterConfig,
    /// Network availability flag, set by the host app (e.g. via NWPathMonitor).
    network_available: AtomicBool,
    /// Last routing decision for introspection.
    last_decision: Mutex<Option<RoutingDecision>>,
    /// Timestamp of last latency measurement (epoch millis).
    last_latency_check: AtomicU64,
}

impl Router {
    pub fn new(
        config: RouterConfig,
        local: Option<Box<dyn Provider>>,
        self_hosted: Vec<Box<dyn Provider>>,
        cloud: Vec<Box<dyn Provider>>,
    ) -> Self {
        let self_hosted = self_hosted
            .into_iter()
            .map(|p| ProviderWithMeta {
                provider: p,
                tier: ProviderTier::SelfHosted,
                latency_ms: AtomicU32::new(u32::MAX),
            })
            .collect();
        let cloud = cloud
            .into_iter()
            .map(|p| ProviderWithMeta {
                provider: p,
                tier: ProviderTier::Cloud,
                latency_ms: AtomicU32::new(u32::MAX),
            })
            .collect();
        Self {
            local,
            self_hosted,
            cloud,
            config,
            network_available: AtomicBool::new(true),
            last_decision: Mutex::new(None),
            last_latency_check: AtomicU64::new(0),
        }
    }

    /// Set network availability (call from Swift when NWPathMonitor reports changes).
    pub fn set_network_available(&self, available: bool) {
        self.network_available.store(available, Ordering::Release);
    }

    /// Get the routing decision from the last `complete()` or `stream()` call.
    pub fn last_routing_decision(&self) -> Option<RoutingDecision> {
        self.last_decision.lock().unwrap().clone()
    }

    // -----------------------------------------------------------------------
    // Latency measurement
    // -----------------------------------------------------------------------

    /// Refresh latency measurements for self-hosted providers if stale.
    fn refresh_latencies(&self) {
        if self.self_hosted.is_empty() {
            return;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let last = self.last_latency_check.load(Ordering::Relaxed);
        if now.saturating_sub(last) < LATENCY_STALE_MS {
            return;
        }
        for sh in &self.self_hosted {
            let start = Instant::now();
            let _ = sh.provider.is_available();
            let ms = start.elapsed().as_millis().min(u32::MAX as u128) as u32;
            sh.latency_ms.store(ms, Ordering::Relaxed);
        }
        self.last_latency_check.store(now, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Provider selection helpers
    // -----------------------------------------------------------------------

    /// Find the first available self-hosted provider, sorted by latency (ascending).
    fn first_available_self_hosted(&self) -> Option<(&dyn Provider, u32)> {
        let mut candidates: Vec<(&ProviderWithMeta, u32)> = self
            .self_hosted
            .iter()
            .filter(|p| p.provider.is_available())
            .map(|p| (p, p.latency_ms.load(Ordering::Relaxed)))
            .collect();
        candidates.sort_by_key(|&(_, lat)| lat);
        candidates
            .first()
            .map(|(p, lat)| (p.provider.as_ref(), *lat))
    }

    fn first_available_cloud(&self) -> Option<&dyn Provider> {
        self.cloud
            .iter()
            .find(|p| p.provider.is_available())
            .map(|p| p.provider.as_ref())
    }

    fn cloud_allowed(&self) -> bool {
        !self.config.privacy_mode && !self.config.data_sovereignty
    }

    // -----------------------------------------------------------------------
    // Provider selection
    // -----------------------------------------------------------------------

    /// Select which provider to use for the current request.
    fn select_provider(&self) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        self.refresh_latencies();

        let network = self.network_available.load(Ordering::Acquire);
        let pressure = quick_check_pressure();

        if self.config.privacy_mode {
            return self.select_local_only(pressure, network);
        }

        match self.config.policy {
            RoutingPolicy::LocalOnly => self.select_local_only(pressure, network),
            RoutingPolicy::LocalFirst => self.select_local_first(pressure, network),
            RoutingPolicy::Auto => self.select_auto(pressure, network),
            RoutingPolicy::CloudFirst => self.select_cloud_first(pressure, network),
            RoutingPolicy::SelfHostedFirst => self.select_self_hosted_first(pressure, network),
        }
    }

    fn make_decision(
        &self,
        provider_name: &str,
        reason: RoutingReason,
        tried: u32,
        tier: ProviderTier,
        pressure: MemoryPressure,
        network: bool,
        latency_ms: u32,
    ) -> RoutingDecision {
        RoutingDecision {
            provider_name: provider_name.to_string(),
            reason,
            providers_tried: tried,
            is_local: tier == ProviderTier::Local,
            provider_tier: tier,
            memory_pressure: pressure,
            network_available: network,
            latency_ms,
        }
    }

    fn select_local_only(
        &self,
        pressure: MemoryPressure,
        network: bool,
    ) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        match &self.local {
            Some(local) if local.is_available() => {
                let reason = if self.config.privacy_mode {
                    RoutingReason::PrivacyMode
                } else {
                    RoutingReason::LocalAvailable
                };
                Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), reason, 1, ProviderTier::Local, pressure, network, 0),
                ))
            }
            _ => Err(CrabInferError::ProviderNotAvailable {
                provider: "local".to_string(),
            }),
        }
    }

    fn select_local_first(
        &self,
        pressure: MemoryPressure,
        network: bool,
    ) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        // 1. If local is available and memory is Normal → local
        if let Some(ref local) = self.local {
            if local.is_available() {
                if pressure == MemoryPressure::Normal {
                    return Ok((
                        local.as_ref(),
                        self.make_decision(local.name(), RoutingReason::LocalAvailable, 1, ProviderTier::Local, pressure, network, 0),
                    ));
                }

                // 2. Memory pressure — try self-hosted first, then cloud
                if let Some((sh, lat)) = self.first_available_self_hosted() {
                    return Ok((
                        sh,
                        self.make_decision(sh.name(), RoutingReason::MemoryPressureFallback, 2, ProviderTier::SelfHosted, pressure, network, lat),
                    ));
                }

                if network && self.cloud_allowed() {
                    if let Some(cloud) = self.first_available_cloud() {
                        return Ok((
                            cloud,
                            self.make_decision(cloud.name(), RoutingReason::MemoryPressureFallback, 2, ProviderTier::Cloud, pressure, network, 0),
                        ));
                    }
                }

                // No fallback available — use local anyway
                return Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), RoutingReason::NetworkUnavailable, 1, ProviderTier::Local, pressure, network, 0),
                ));
            }
        }

        // 3. No local model — try self-hosted, then cloud
        if let Some((sh, lat)) = self.first_available_self_hosted() {
            return Ok((
                sh,
                self.make_decision(sh.name(), RoutingReason::NoLocalModel, 1, ProviderTier::SelfHosted, pressure, network, lat),
            ));
        }

        if network && self.cloud_allowed() {
            if let Some(cloud) = self.first_available_cloud() {
                return Ok((
                    cloud,
                    self.make_decision(cloud.name(), RoutingReason::NoLocalModel, 1, ProviderTier::Cloud, pressure, network, 0),
                ));
            }
        }

        Err(CrabInferError::ProviderNotAvailable {
            provider: "no provider available".to_string(),
        })
    }

    fn select_auto(
        &self,
        pressure: MemoryPressure,
        network: bool,
    ) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        // Auto: local (if normal memory) → self-hosted (by latency) → cloud

        // 1. If local available and normal memory → local
        if let Some(ref local) = self.local {
            if local.is_available() && pressure == MemoryPressure::Normal {
                return Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), RoutingReason::LocalAvailable, 1, ProviderTier::Local, pressure, network, 0),
                ));
            }
        }

        // 2. Try self-hosted (sorted by latency)
        if let Some((sh, lat)) = self.first_available_self_hosted() {
            let reason = if self.local.is_some() {
                RoutingReason::MemoryPressureFallback
            } else {
                RoutingReason::SelfHostedAvailable
            };
            return Ok((
                sh,
                self.make_decision(sh.name(), reason, 2, ProviderTier::SelfHosted, pressure, network, lat),
            ));
        }

        // 3. Try cloud (if allowed)
        if network && self.cloud_allowed() {
            if let Some(cloud) = self.first_available_cloud() {
                return Ok((
                    cloud,
                    self.make_decision(cloud.name(), RoutingReason::NoLocalModel, 2, ProviderTier::Cloud, pressure, network, 0),
                ));
            }
        }

        // 4. Fall back to local even under pressure
        if let Some(ref local) = self.local {
            if local.is_available() {
                return Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), RoutingReason::NetworkUnavailable, 1, ProviderTier::Local, pressure, network, 0),
                ));
            }
        }

        Err(CrabInferError::ProviderNotAvailable {
            provider: "no provider available".to_string(),
        })
    }

    fn select_cloud_first(
        &self,
        pressure: MemoryPressure,
        network: bool,
    ) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        // Cloud first → self-hosted → local

        // 1. Try cloud
        if network && self.cloud_allowed() {
            if let Some(cloud) = self.first_available_cloud() {
                return Ok((
                    cloud,
                    self.make_decision(cloud.name(), RoutingReason::CloudPreferred, 1, ProviderTier::Cloud, pressure, network, 0),
                ));
            }
        }

        // 2. Try self-hosted
        if let Some((sh, lat)) = self.first_available_self_hosted() {
            return Ok((
                sh,
                self.make_decision(sh.name(), RoutingReason::SelfHostedAvailable, 1, ProviderTier::SelfHosted, pressure, network, lat),
            ));
        }

        // 3. Fall back to local
        if let Some(ref local) = self.local {
            if local.is_available() {
                return Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), RoutingReason::NetworkUnavailable, 1, ProviderTier::Local, pressure, network, 0),
                ));
            }
        }

        Err(CrabInferError::ProviderNotAvailable {
            provider: "no provider available".to_string(),
        })
    }

    fn select_self_hosted_first(
        &self,
        pressure: MemoryPressure,
        network: bool,
    ) -> Result<(&dyn Provider, RoutingDecision), CrabInferError> {
        // Self-hosted first → local → cloud

        // 1. Try self-hosted (by latency)
        if let Some((sh, lat)) = self.first_available_self_hosted() {
            return Ok((
                sh,
                self.make_decision(sh.name(), RoutingReason::SelfHostedPreferred, 1, ProviderTier::SelfHosted, pressure, network, lat),
            ));
        }

        // 2. Fall back to local
        if let Some(ref local) = self.local {
            if local.is_available() {
                return Ok((
                    local.as_ref(),
                    self.make_decision(local.name(), RoutingReason::LocalAvailable, 1, ProviderTier::Local, pressure, network, 0),
                ));
            }
        }

        // 3. Fall back to cloud (if allowed)
        if network && self.cloud_allowed() {
            if let Some(cloud) = self.first_available_cloud() {
                return Ok((
                    cloud,
                    self.make_decision(cloud.name(), RoutingReason::CloudPreferred, 1, ProviderTier::Cloud, pressure, network, 0),
                ));
            }
        }

        Err(CrabInferError::ProviderNotAvailable {
            provider: "no provider available".to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Fallback logic (3-tier cascade)
    // -----------------------------------------------------------------------

    fn try_providers_for_complete(
        &self,
        request: &CompletionRequest,
        original: &RoutingDecision,
        skip_provider: &str,
    ) -> Option<CompletionResponse> {
        let tried = original.providers_tried;

        // Try other providers in the same tier first (skip only the one that failed)
        match original.provider_tier {
            ProviderTier::Cloud => {
                if self.cloud_allowed() {
                    for c in &self.cloud {
                        if c.provider.name() != skip_provider && c.provider.is_available() {
                            if let Ok(mut resp) = c.provider.complete(request) {
                                let decision = self.make_decision(
                                    c.provider.name(),
                                    RoutingReason::FallbackAfterError,
                                    tried + 1,
                                    ProviderTier::Cloud,
                                    original.memory_pressure.clone(),
                                    original.network_available,
                                    0,
                                );
                                resp.routing_info = format_routing_info(&decision);
                                *self.last_decision.lock().unwrap() = Some(decision);
                                return Some(resp);
                            }
                        }
                    }
                }
            }
            ProviderTier::SelfHosted => {
                for sh in &self.self_hosted {
                    if sh.provider.name() != skip_provider && sh.provider.is_available() {
                        if let Ok(mut resp) = sh.provider.complete(request) {
                            let decision = self.make_decision(
                                sh.provider.name(),
                                RoutingReason::FallbackAfterError,
                                tried + 1,
                                ProviderTier::SelfHosted,
                                original.memory_pressure.clone(),
                                original.network_available,
                                sh.latency_ms.load(Ordering::Relaxed),
                            );
                            resp.routing_info = format_routing_info(&decision);
                            *self.last_decision.lock().unwrap() = Some(decision);
                            return Some(resp);
                        }
                    }
                }
            }
            ProviderTier::Local => {} // Only one local provider
        }

        // Try self-hosted (if original tier was not self-hosted)
        if original.provider_tier != ProviderTier::SelfHosted {
            for sh in &self.self_hosted {
                if sh.provider.is_available() {
                    if let Ok(mut resp) = sh.provider.complete(request) {
                        let decision = self.make_decision(
                            sh.provider.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::SelfHosted,
                            original.memory_pressure.clone(),
                            original.network_available,
                            sh.latency_ms.load(Ordering::Relaxed),
                        );
                        resp.routing_info = format_routing_info(&decision);
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(resp);
                    }
                }
            }
        }

        // Try cloud (if allowed and original tier was not cloud)
        if original.provider_tier != ProviderTier::Cloud && self.cloud_allowed() {
            for c in &self.cloud {
                if c.provider.is_available() {
                    if let Ok(mut resp) = c.provider.complete(request) {
                        let decision = self.make_decision(
                            c.provider.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::Cloud,
                            original.memory_pressure.clone(),
                            original.network_available,
                            0,
                        );
                        resp.routing_info = format_routing_info(&decision);
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(resp);
                    }
                }
            }
        }

        // Try local (if original tier was not local)
        if original.provider_tier != ProviderTier::Local {
            if let Some(ref local) = self.local {
                if local.is_available() {
                    if let Ok(mut resp) = local.complete(request) {
                        let decision = self.make_decision(
                            local.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::Local,
                            original.memory_pressure.clone(),
                            original.network_available,
                            0,
                        );
                        resp.routing_info = format_routing_info(&decision);
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(resp);
                    }
                }
            }
        }

        None
    }

    fn complete_with_fallback(
        &self,
        request: &CompletionRequest,
        original: &RoutingDecision,
        err: CrabInferError,
    ) -> Result<CompletionResponse, CrabInferError> {
        if let Some(resp) = self.try_providers_for_complete(request, original, &original.provider_name) {
            return Ok(resp);
        }
        Err(err)
    }

    fn try_providers_for_stream(
        &self,
        request: &CompletionRequest,
        original: &RoutingDecision,
        skip_provider: &str,
    ) -> Option<Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>> {
        let tried = original.providers_tried;

        // Try other providers in the same tier first (skip only the one that failed)
        match original.provider_tier {
            ProviderTier::Cloud => {
                if self.cloud_allowed() {
                    for c in &self.cloud {
                        if c.provider.name() != skip_provider && c.provider.is_available() {
                            if let Ok(iter) = c.provider.stream(request) {
                                let decision = self.make_decision(
                                    c.provider.name(),
                                    RoutingReason::FallbackAfterError,
                                    tried + 1,
                                    ProviderTier::Cloud,
                                    original.memory_pressure.clone(),
                                    original.network_available,
                                    0,
                                );
                                *self.last_decision.lock().unwrap() = Some(decision);
                                return Some(iter);
                            }
                        }
                    }
                }
            }
            ProviderTier::SelfHosted => {
                for sh in &self.self_hosted {
                    if sh.provider.name() != skip_provider && sh.provider.is_available() {
                        if let Ok(iter) = sh.provider.stream(request) {
                            let decision = self.make_decision(
                                sh.provider.name(),
                                RoutingReason::FallbackAfterError,
                                tried + 1,
                                ProviderTier::SelfHosted,
                                original.memory_pressure.clone(),
                                original.network_available,
                                sh.latency_ms.load(Ordering::Relaxed),
                            );
                            *self.last_decision.lock().unwrap() = Some(decision);
                            return Some(iter);
                        }
                    }
                }
            }
            ProviderTier::Local => {} // Only one local provider
        }

        // Try self-hosted (if original tier was not self-hosted)
        if original.provider_tier != ProviderTier::SelfHosted {
            for sh in &self.self_hosted {
                if sh.provider.is_available() {
                    if let Ok(iter) = sh.provider.stream(request) {
                        let decision = self.make_decision(
                            sh.provider.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::SelfHosted,
                            original.memory_pressure.clone(),
                            original.network_available,
                            sh.latency_ms.load(Ordering::Relaxed),
                        );
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(iter);
                    }
                }
            }
        }

        // Try cloud (if allowed and original tier was not cloud)
        if original.provider_tier != ProviderTier::Cloud && self.cloud_allowed() {
            for c in &self.cloud {
                if c.provider.is_available() {
                    if let Ok(iter) = c.provider.stream(request) {
                        let decision = self.make_decision(
                            c.provider.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::Cloud,
                            original.memory_pressure.clone(),
                            original.network_available,
                            0,
                        );
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(iter);
                    }
                }
            }
        }

        // Try local (if original tier was not local)
        if original.provider_tier != ProviderTier::Local {
            if let Some(ref local) = self.local {
                if local.is_available() {
                    if let Ok(iter) = local.stream(request) {
                        let decision = self.make_decision(
                            local.name(),
                            RoutingReason::FallbackAfterError,
                            tried + 1,
                            ProviderTier::Local,
                            original.memory_pressure.clone(),
                            original.network_available,
                            0,
                        );
                        *self.last_decision.lock().unwrap() = Some(decision);
                        return Some(iter);
                    }
                }
            }
        }

        None
    }

    fn stream_with_fallback(
        &self,
        request: &CompletionRequest,
        original: &RoutingDecision,
        err: CrabInferError,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>,
        CrabInferError,
    > {
        if let Some(iter) = self.try_providers_for_stream(request, original, &original.provider_name) {
            return Ok(iter);
        }
        Err(err)
    }
}

// ---------------------------------------------------------------------------
// Provider trait implementation
// ---------------------------------------------------------------------------

impl Provider for Router {
    fn name(&self) -> &str {
        "router"
    }

    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        let (provider, decision) = self.select_provider()?;
        let routing_info = format_routing_info(&decision);
        *self.last_decision.lock().unwrap() = Some(decision.clone());

        match provider.complete(request) {
            Ok(mut resp) => {
                resp.routing_info = routing_info;
                Ok(resp)
            }
            Err(e) => self.complete_with_fallback(request, &decision, e),
        }
    }

    fn stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>,
        CrabInferError,
    > {
        let (provider, decision) = self.select_provider()?;
        *self.last_decision.lock().unwrap() = Some(decision.clone());

        match provider.stream(request) {
            Ok(iter) => Ok(iter),
            Err(e) => self.stream_with_fallback(request, &decision, e),
        }
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        let mut models = Vec::new();
        if let Some(ref local) = self.local {
            if let Ok(mut m) = local.available_models() {
                models.append(&mut m);
            }
        }
        for sh in &self.self_hosted {
            if let Ok(mut m) = sh.provider.available_models() {
                models.append(&mut m);
            }
        }
        for cloud in &self.cloud {
            if let Ok(mut m) = cloud.provider.available_models() {
                models.append(&mut m);
            }
        }
        Ok(models)
    }

    fn is_available(&self) -> bool {
        let local_ok = self.local.as_ref().map_or(false, |p| p.is_available());
        let sh_ok = self.self_hosted.iter().any(|p| p.provider.is_available());
        let cloud_ok = self.cloud.iter().any(|p| p.provider.is_available());
        local_ok || sh_ok || cloud_ok
    }
}

// ---------------------------------------------------------------------------
// Tier resolution helpers
// ---------------------------------------------------------------------------

/// Resolve the tier for a provider based on its config.
pub fn resolve_tier(config: &crate::provider::ProviderConfig) -> ProviderTier {
    if !config.tier_override.is_empty() {
        return match config.tier_override.as_str() {
            "self_hosted" | "selfhosted" => ProviderTier::SelfHosted,
            "cloud" => ProviderTier::Cloud,
            "local" => ProviderTier::Local,
            _ => default_tier_for_provider(&config.provider_type),
        };
    }
    default_tier_for_provider(&config.provider_type)
}

/// Default tier classification based on provider type.
pub fn default_tier_for_provider(provider_type: &str) -> ProviderTier {
    match provider_type {
        "ollama" | "vllm" => ProviderTier::SelfHosted,
        _ => ProviderTier::Cloud,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_routing_info(decision: &RoutingDecision) -> String {
    format!(
        "{} (tier: {:?}, reason: {:?}, pressure: {:?}, network: {}, latency: {}ms)",
        decision.provider_name,
        decision.provider_tier,
        decision.reason,
        decision.memory_pressure,
        if decision.network_available { "yes" } else { "no" },
        decision.latency_ms,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatMessage;

    struct MockProvider {
        provider_name: &'static str,
        available: bool,
        should_fail: bool,
    }

    impl Provider for MockProvider {
        fn name(&self) -> &str { self.provider_name }
        fn complete(&self, _request: &CompletionRequest) -> Result<CompletionResponse, CrabInferError> {
            if self.should_fail {
                return Err(CrabInferError::NetworkError { reason: "mock failure".to_string() });
            }
            Ok(CompletionResponse {
                content: format!("from {}", self.provider_name),
                model: "mock-model".to_string(),
                provider_name: self.provider_name.to_string(),
                stop_reason: "end_turn".to_string(),
                input_tokens: 10,
                output_tokens: 5,
                routing_info: String::new(),
            })
        }
        fn stream(&self, _request: &CompletionRequest) -> Result<Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>, CrabInferError> {
            if self.should_fail {
                return Err(CrabInferError::NetworkError { reason: "mock failure".to_string() });
            }
            Ok(Box::new(vec![Ok(TokenOutput { text: "hello".to_string(), token_id: 1, probability: 1.0, is_end_of_sequence: true })].into_iter()))
        }
        fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
            Ok(vec![ModelDescriptor { id: format!("{}-model", self.provider_name), name: format!("{} Model", self.provider_name), provider: self.provider_name.to_string(), is_local: self.provider_name == "local", context_length: 4096 }])
        }
        fn is_available(&self) -> bool { self.available }
    }

    fn mock_local(available: bool, should_fail: bool) -> Box<dyn Provider> {
        Box::new(MockProvider { provider_name: "local", available, should_fail })
    }
    fn mock_self_hosted(name: &'static str, available: bool, should_fail: bool) -> Box<dyn Provider> {
        Box::new(MockProvider { provider_name: name, available, should_fail })
    }
    fn mock_cloud(name: &'static str, available: bool, should_fail: bool) -> Box<dyn Provider> {
        Box::new(MockProvider { provider_name: name, available, should_fail })
    }

    fn test_request() -> CompletionRequest {
        CompletionRequest { model: "test".to_string(), messages: vec![ChatMessage { role: "user".to_string(), content: "hello".to_string() }], max_tokens: 100, temperature: 0.7, top_p: 0.9, system_prompt: String::new(), api_key_override: String::new() }
    }

    fn config(policy: RoutingPolicy) -> RouterConfig {
        RouterConfig { policy, privacy_mode: false, ollama_is_local: true, data_sovereignty: false }
    }

    // --- Existing tests (updated for 3-tier API) ---

    #[test]
    fn test_local_first_normal_memory_routes_local() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        let resp = router.complete(&test_request()).unwrap();
        assert_eq!(resp.provider_name, "local");
        let d = router.last_routing_decision().unwrap();
        assert_eq!(d.reason, RoutingReason::LocalAvailable);
        assert!(d.is_local);
        assert_eq!(d.provider_tier, ProviderTier::Local);
    }

    #[test]
    fn test_auto_prefers_local_normal_memory() {
        let router = Router::new(config(RoutingPolicy::Auto), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
    }

    #[test]
    fn test_local_first_no_local_routes_cloud() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), None, vec![], vec![mock_cloud("openai", true, false)]);
        let resp = router.complete(&test_request()).unwrap();
        assert_eq!(resp.provider_name, "openai");
        let d = router.last_routing_decision().unwrap();
        assert_eq!(d.reason, RoutingReason::NoLocalModel);
        assert!(!d.is_local);
        assert_eq!(d.provider_tier, ProviderTier::Cloud);
    }

    #[test]
    fn test_local_first_local_unavailable_routes_cloud() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(false, false)), vec![], vec![mock_cloud("anthropic", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "anthropic");
    }

    #[test]
    fn test_cloud_first_routes_to_cloud() {
        let router = Router::new(config(RoutingPolicy::CloudFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        let resp = router.complete(&test_request()).unwrap();
        assert_eq!(resp.provider_name, "openai");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::CloudPreferred);
    }

    #[test]
    fn test_cloud_first_no_network_falls_back_to_local() {
        let router = Router::new(config(RoutingPolicy::CloudFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        router.set_network_available(false);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::NetworkUnavailable);
    }

    #[test]
    fn test_local_only_never_uses_cloud() {
        let router = Router::new(config(RoutingPolicy::LocalOnly), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
    }

    #[test]
    fn test_local_only_no_local_returns_error() {
        let router = Router::new(config(RoutingPolicy::LocalOnly), None, vec![], vec![mock_cloud("openai", true, false)]);
        assert!(matches!(router.complete(&test_request()).unwrap_err(), CrabInferError::ProviderNotAvailable { .. }));
    }

    #[test]
    fn test_privacy_mode_blocks_cloud() {
        let router = Router::new(RouterConfig { policy: RoutingPolicy::CloudFirst, privacy_mode: true, ollama_is_local: true, data_sovereignty: false }, Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::PrivacyMode);
    }

    #[test]
    fn test_fallback_after_local_error() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, true)), vec![], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "openai");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::FallbackAfterError);
    }

    #[test]
    fn test_fallback_after_cloud_error() {
        let router = Router::new(config(RoutingPolicy::CloudFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, true)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::FallbackAfterError);
    }

    #[test]
    fn test_no_providers_returns_error() {
        let router = Router::new(config(RoutingPolicy::Auto), None, vec![], vec![]);
        assert!(matches!(router.complete(&test_request()).unwrap_err(), CrabInferError::ProviderNotAvailable { .. }));
    }

    #[test]
    fn test_available_models_aggregated() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false), mock_cloud("anthropic", true, false)]);
        let models = router.available_models().unwrap();
        assert_eq!(models.len(), 4);
        let providers: Vec<&str> = models.iter().map(|m| m.provider.as_str()).collect();
        assert!(providers.contains(&"local"));
        assert!(providers.contains(&"ollama"));
        assert!(providers.contains(&"openai"));
        assert!(providers.contains(&"anthropic"));
    }

    #[test]
    fn test_routing_decision_stored() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![], vec![]);
        assert!(router.last_routing_decision().is_none());
        let _ = router.complete(&test_request()).unwrap();
        assert!(router.last_routing_decision().is_some());
    }

    #[test]
    fn test_is_available() {
        let router_all = Router::new(config(RoutingPolicy::Auto), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert!(router_all.is_available());
        let router_none = Router::new(config(RoutingPolicy::Auto), Some(mock_local(false, false)), vec![mock_self_hosted("ollama", false, false)], vec![mock_cloud("openai", false, false)]);
        assert!(!router_none.is_available());
    }

    #[test]
    fn test_stream_routes_correctly() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        let mut iter = router.stream(&test_request()).unwrap();
        let token = iter.next().unwrap().unwrap();
        assert_eq!(token.text, "hello");
        assert!(router.last_routing_decision().unwrap().is_local);
    }

    #[test]
    fn test_stream_fallback_after_error() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, true)), vec![], vec![mock_cloud("openai", true, false)]);
        let mut iter = router.stream(&test_request()).unwrap();
        assert_eq!(iter.next().unwrap().unwrap().text, "hello");
        assert_eq!(router.last_routing_decision().unwrap().reason, RoutingReason::FallbackAfterError);
    }

    #[test]
    fn test_local_first_no_network_no_cloud_still_uses_local() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        router.set_network_available(false);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
    }

    // --- New 3-tier tests ---

    #[test]
    fn test_auto_three_tier_cascade() {
        let router = Router::new(config(RoutingPolicy::Auto), None, vec![mock_self_hosted("vllm", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "vllm");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_auto_no_local_no_self_hosted_routes_cloud() {
        let router = Router::new(config(RoutingPolicy::Auto), None, vec![], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "openai");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::Cloud);
    }

    #[test]
    fn test_self_hosted_first_routes_to_self_hosted() {
        let router = Router::new(config(RoutingPolicy::SelfHostedFirst), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "ollama");
        let d = router.last_routing_decision().unwrap();
        assert_eq!(d.reason, RoutingReason::SelfHostedPreferred);
        assert_eq!(d.provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_self_hosted_first_falls_back_to_local() {
        let router = Router::new(config(RoutingPolicy::SelfHostedFirst), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", false, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
    }

    #[test]
    fn test_data_sovereignty_blocks_cloud() {
        let router = Router::new(RouterConfig { policy: RoutingPolicy::Auto, privacy_mode: false, ollama_is_local: true, data_sovereignty: true }, None, vec![mock_self_hosted("vllm", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "vllm");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_data_sovereignty_no_self_hosted_returns_error() {
        let router = Router::new(RouterConfig { policy: RoutingPolicy::Auto, privacy_mode: false, ollama_is_local: true, data_sovereignty: true }, None, vec![], vec![mock_cloud("openai", true, false)]);
        assert!(matches!(router.complete(&test_request()).unwrap_err(), CrabInferError::ProviderNotAvailable { .. }));
    }

    #[test]
    fn test_fallback_local_to_self_hosted() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, true)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "ollama");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_fallback_respects_data_sovereignty() {
        // Self-hosted errors, cloud available but blocked by data_sovereignty → original error propagated
        let router = Router::new(RouterConfig { policy: RoutingPolicy::SelfHostedFirst, privacy_mode: false, ollama_is_local: true, data_sovereignty: true }, None, vec![mock_self_hosted("vllm", true, true)], vec![mock_cloud("openai", true, false)]);
        let err = router.complete(&test_request()).unwrap_err();
        // Fallback exhausted: original error from self-hosted provider is returned (not ProviderNotAvailable)
        assert!(matches!(err, CrabInferError::NetworkError { .. }));
    }

    #[test]
    fn test_self_hosted_models_aggregated() {
        let router = Router::new(config(RoutingPolicy::Auto), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false), mock_self_hosted("vllm", true, false)], vec![mock_cloud("openai", true, false)]);
        let models = router.available_models().unwrap();
        assert_eq!(models.len(), 4);
        let providers: Vec<&str> = models.iter().map(|m| m.provider.as_str()).collect();
        assert!(providers.contains(&"ollama"));
        assert!(providers.contains(&"vllm"));
    }

    #[test]
    fn test_is_available_three_tiers() {
        let router = Router::new(config(RoutingPolicy::Auto), Some(mock_local(false, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", false, false)]);
        assert!(router.is_available());
    }

    #[test]
    fn test_backward_compat_empty_self_hosted() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), Some(mock_local(true, false)), vec![], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "local");
    }

    #[test]
    fn test_stream_through_self_hosted() {
        let router = Router::new(config(RoutingPolicy::SelfHostedFirst), Some(mock_local(true, false)), vec![mock_self_hosted("vllm", true, false)], vec![]);
        let mut iter = router.stream(&test_request()).unwrap();
        assert_eq!(iter.next().unwrap().unwrap().text, "hello");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_network_unavailable_allows_self_hosted() {
        let router = Router::new(config(RoutingPolicy::CloudFirst), Some(mock_local(true, false)), vec![mock_self_hosted("ollama", true, false)], vec![mock_cloud("openai", true, false)]);
        router.set_network_available(false);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "ollama");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_local_first_no_local_prefers_self_hosted() {
        let router = Router::new(config(RoutingPolicy::LocalFirst), None, vec![mock_self_hosted("vllm", true, false)], vec![mock_cloud("openai", true, false)]);
        assert_eq!(router.complete(&test_request()).unwrap().provider_name, "vllm");
        assert_eq!(router.last_routing_decision().unwrap().provider_tier, ProviderTier::SelfHosted);
    }

    #[test]
    fn test_tier_resolution() {
        use crate::provider::ProviderConfig;
        let base = ProviderConfig { provider_type: "ollama".to_string(), api_key: String::new(), base_url: String::new(), default_model: String::new(), timeout_seconds: 0, tier_override: String::new() };
        assert_eq!(resolve_tier(&base), ProviderTier::SelfHosted);
        assert_eq!(resolve_tier(&ProviderConfig { provider_type: "openai".to_string(), ..base.clone() }), ProviderTier::Cloud);
        assert_eq!(resolve_tier(&ProviderConfig { provider_type: "openai".to_string(), tier_override: "self_hosted".to_string(), ..base.clone() }), ProviderTier::SelfHosted);
        assert_eq!(default_tier_for_provider("vllm"), ProviderTier::SelfHosted);
        assert_eq!(default_tier_for_provider("anthropic"), ProviderTier::Cloud);
        assert_eq!(default_tier_for_provider("google"), ProviderTier::Cloud);
    }
}
