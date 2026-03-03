use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Server error type that converts CrabInferError into HTTP responses
#[derive(Debug)]
pub struct ServerError {
    pub status: StatusCode,
    pub message: String,
}

impl ServerError {
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn service_unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: msg.into(),
        }
    }

    pub fn too_many_requests(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": "server_error",
                "code": self.status.as_u16(),
            }
        });
        (self.status, Json(body)).into_response()
    }
}

impl From<crabinfer_core::CrabInferError> for ServerError {
    fn from(err: crabinfer_core::CrabInferError) -> Self {
        use crabinfer_core::CrabInferError;
        match &err {
            CrabInferError::ModelNotFound => ServerError::bad_request(err.to_string()),
            CrabInferError::InvalidConfig => ServerError::bad_request(err.to_string()),
            CrabInferError::ContextOverflow => ServerError::bad_request(err.to_string()),
            CrabInferError::OutOfMemory { .. } => ServerError::service_unavailable(err.to_string()),
            CrabInferError::ModelTooLarge { .. } => ServerError::bad_request(err.to_string()),
            _ => ServerError::internal(err.to_string()),
        }
    }
}
