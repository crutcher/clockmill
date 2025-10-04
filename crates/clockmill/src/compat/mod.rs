//! # Compat Mechanisms for upcoming Burn API Changes
pub mod operations;

/// `1.0 / (3.0).sqrt()`
/// TODO: unstable feature: f64::FRAC_1_SQRT_3
pub const FRAC_1_SQRT_3: f64 = 0.577350269189625764509148780501957456_f64;