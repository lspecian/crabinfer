//! Enum types mapped from crabinfer-core to napi.

use napi_derive::napi;

/// Memory pressure levels.
#[napi(string_enum)]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
    Terminal,
}

impl From<crabinfer_core::MemoryPressure> for MemoryPressure {
    fn from(p: crabinfer_core::MemoryPressure) -> Self {
        match p {
            crabinfer_core::MemoryPressure::Normal => Self::Normal,
            crabinfer_core::MemoryPressure::Warning => Self::Warning,
            crabinfer_core::MemoryPressure::Critical => Self::Critical,
            crabinfer_core::MemoryPressure::Terminal => Self::Terminal,
        }
    }
}

/// Provider infrastructure tier.
#[napi(string_enum)]
pub enum ProviderTier {
    Local,
    SelfHosted,
    Cloud,
}

impl From<crabinfer_core::router::ProviderTier> for ProviderTier {
    fn from(t: crabinfer_core::router::ProviderTier) -> Self {
        match t {
            crabinfer_core::router::ProviderTier::Local => Self::Local,
            crabinfer_core::router::ProviderTier::SelfHosted => Self::SelfHosted,
            crabinfer_core::router::ProviderTier::Cloud => Self::Cloud,
        }
    }
}

/// Routing policy.
#[napi(string_enum)]
pub enum RoutingPolicy {
    LocalFirst,
    CloudFirst,
    LocalOnly,
    Auto,
    SelfHostedFirst,
}

impl From<RoutingPolicy> for crabinfer_core::router::RoutingPolicy {
    fn from(p: RoutingPolicy) -> Self {
        match p {
            RoutingPolicy::LocalFirst => Self::LocalFirst,
            RoutingPolicy::CloudFirst => Self::CloudFirst,
            RoutingPolicy::LocalOnly => Self::LocalOnly,
            RoutingPolicy::Auto => Self::Auto,
            RoutingPolicy::SelfHostedFirst => Self::SelfHostedFirst,
        }
    }
}

/// Routing reason — why a provider was selected.
#[napi(string_enum)]
pub enum RoutingReason {
    LocalAvailable,
    MemoryPressureFallback,
    NoLocalModel,
    CloudPreferred,
    NetworkUnavailable,
    PrivacyMode,
    FallbackAfterError,
    SelfHostedAvailable,
    SelfHostedPreferred,
    DataSovereignty,
}

impl From<crabinfer_core::router::RoutingReason> for RoutingReason {
    fn from(r: crabinfer_core::router::RoutingReason) -> Self {
        match r {
            crabinfer_core::router::RoutingReason::LocalAvailable => Self::LocalAvailable,
            crabinfer_core::router::RoutingReason::MemoryPressureFallback => {
                Self::MemoryPressureFallback
            }
            crabinfer_core::router::RoutingReason::NoLocalModel => Self::NoLocalModel,
            crabinfer_core::router::RoutingReason::CloudPreferred => Self::CloudPreferred,
            crabinfer_core::router::RoutingReason::NetworkUnavailable => Self::NetworkUnavailable,
            crabinfer_core::router::RoutingReason::PrivacyMode => Self::PrivacyMode,
            crabinfer_core::router::RoutingReason::FallbackAfterError => Self::FallbackAfterError,
            crabinfer_core::router::RoutingReason::SelfHostedAvailable => {
                Self::SelfHostedAvailable
            }
            crabinfer_core::router::RoutingReason::SelfHostedPreferred => {
                Self::SelfHostedPreferred
            }
            crabinfer_core::router::RoutingReason::DataSovereignty => Self::DataSovereignty,
        }
    }
}

/// Model use-case category.
#[napi(string_enum)]
pub enum ModelCategory {
    General,
    Code,
    Reasoning,
}

impl From<crabinfer_core::catalog::ModelCategory> for ModelCategory {
    fn from(c: crabinfer_core::catalog::ModelCategory) -> Self {
        match c {
            crabinfer_core::catalog::ModelCategory::General => Self::General,
            crabinfer_core::catalog::ModelCategory::Code => Self::Code,
            crabinfer_core::catalog::ModelCategory::Reasoning => Self::Reasoning,
        }
    }
}

/// Download status.
#[cfg(feature = "providers")]
#[napi(string_enum)]
pub enum DownloadStatus {
    NotDownloaded,
    Downloading,
    Downloaded,
    Failed,
}

#[cfg(feature = "providers")]
impl From<crabinfer_core::download::DownloadStatus> for DownloadStatus {
    fn from(s: crabinfer_core::download::DownloadStatus) -> Self {
        match s {
            crabinfer_core::download::DownloadStatus::NotDownloaded => Self::NotDownloaded,
            crabinfer_core::download::DownloadStatus::Downloading => Self::Downloading,
            crabinfer_core::download::DownloadStatus::Downloaded => Self::Downloaded,
            crabinfer_core::download::DownloadStatus::Failed => Self::Failed,
        }
    }
}
