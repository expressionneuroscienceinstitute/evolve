//! Authentication and authorization for God mode access

use axum::{
    async_trait,
    extract::{FromRequestParts, TypedHeader},
    headers::{authorization::Bearer, Authorization},
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
    RequestPartsExt,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// JWT claims for authenticated sessions
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub god_mode: bool,
    pub exp: u64,
}

/// Auth layer for protected routes
pub struct AuthLayer;

#[async_trait]
impl<S> FromRequestParts<S> for Claims
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract the token from the authorization header
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await
            .map_err(|_| AuthError::InvalidToken)?;

        // Decode and validate the token
        let token_data = decode::<Claims>(
            bearer.token(),
            &DecodingKey::from_secret(JWT_SECRET.as_ref()),
            &Validation::default(),
        )
        .map_err(|_| AuthError::InvalidToken)?;

        Ok(token_data.claims)
    }
}

/// Auth errors
#[derive(Debug)]
pub enum AuthError {
    InvalidToken,
    Unauthorized,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AuthError::InvalidToken => (StatusCode::BAD_REQUEST, "Invalid token"),
            AuthError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized"),
        };

        (status, error_message).into_response()
    }
}

// Secret key for JWT signing (in production, load from environment)
const JWT_SECRET: &[u8] = b"your-secret-key-change-in-production";

/// Generate a JWT token for authenticated sessions
pub fn generate_token(username: &str, god_mode: bool) -> Result<String, jsonwebtoken::errors::Error> {
    let expiration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() + 3600; // 1 hour

    let claims = Claims {
        sub: username.to_owned(),
        god_mode,
        exp: expiration,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(JWT_SECRET),
    )
}