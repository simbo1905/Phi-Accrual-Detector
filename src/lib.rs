//!
//! This is a pluggable implementation of the Phi Accrual Failure Detector.
//!
//! The simplest implementation to use it in your system has been shown in the examples/monitor.rs
//!
//! ```rust
//!use phi_accrual_detector::{Detector, PhiInteraction};
//!use async_trait::async_trait;
//!use std::sync::{Arc};
//!use chrono::{DateTime, Local};
//!
//!struct Monitor {
//!    detector: Arc<Detector>,
//!}
//!
//!#[async_trait]
//!trait MonitorInteraction {
//!    // For inserting heartbeat arrival time
//!    async fn ping(&self);
//!    // For calculating suspicion level
//!    async fn suspicion(&self) -> f64;
//!}
//!
//!#[async_trait]
//!impl MonitorInteraction for Monitor {
//!     async fn ping(&self) {
//!         let current_time = Local::now();
//!         self.detector.insert(current_time).await.expect("Some panic occurred");
//!    }
//!
//!    async fn suspicion(&self) -> f64 {
//!        let current_time = Local::now();
//!        let last_arrived_at = self.detector.last_arrived_at().await.expect("Some panic occurred");
//!        // you can determine an acceptable threshold (ex:0.5) for phi after which you can take action.
//!        let phi = self.detector.phi(current_time).await.unwrap();
//!        phi
//!    }
//!}
//!
//! fn main() {
//!   let detector = Arc::new(Detector::new(1000));
//!   let monitor = Monitor { detector: Arc::clone(&detector) };
//! }
//! ```
//!
//! The above example gives you an implementation of a Monitor struct which can be used to interact
//! with the Detector struct. However, if you want to give your process some leeway to recover from
//! a failure or account in the network latencies, you can set an acceptable pause duration during
//! which the detector will not raise suspicion. You can tweak the detector in the above example like
//! this:
//!
//! ```rust
//! use phi_accrual_detector::{Detector};
//! use async_trait::async_trait;
//! use std::sync::{Arc};
//! use chrono::{TimeDelta};
//!
//! struct Monitor {
//!    detector: Arc<Detector>,
//! }
//!
//! // implementation and traits remain the same.
//!
//! fn main() {
//!   let detector = Arc::new(Detector::with_acceptable_pause(1000, TimeDelta::milliseconds(1000)));
//!   let monitor = Monitor { detector: Arc::clone(&detector) };
//! }
//! ```
//!
use std::error::Error;
use std::ops::Sub;
use std::sync::{Arc};
use tokio::sync::{RwLock, RwLockReadGuard};
use async_trait::async_trait;
use libm::{erf, log10};
use chrono::{DateTime, Local, TimeDelta};

/// Statistics of last window_length intervals
#[derive(Clone, Debug)]
pub struct Statistics {
    arrival_intervals: Vec<u64>,
    last_arrived_at: DateTime<Local>,
    window_length: u32,
    n: u32,
}

/// Detector meant for abstraction over Statistics
#[derive(Debug)]
pub struct Detector {
    statistics: RwLock<Statistics>,
    acceptable_pause: TimeDelta,
}

impl Detector {
    /// New Detector instance with window_length. Recommended window_length is < 10000
    pub fn new(window_length: u32) -> Self {
        Detector {
            statistics: RwLock::new(Statistics::new(window_length)),
            acceptable_pause: TimeDelta::milliseconds(0),
        }
    }

    /// New Detector instance with acceptable heartbeat pause duration.
    pub fn with_acceptable_pause(window_length: u32, acceptable_pause: TimeDelta) -> Self {
        Detector {
            statistics: RwLock::new(Statistics::new(window_length)),
            acceptable_pause,
        }
    }
}

impl Statistics {
    /// New Statistics instance with window_length.
    pub fn new(window_length: u32) -> Self {
        Self {
            arrival_intervals: vec![],
            last_arrived_at: Local::now(),
            window_length,
            n: 0,
        }
    }

    /// Insert heartbeat arrival time in window.
    pub fn insert(&mut self, arrived_at: DateTime<Local>) {

        // insert first element
        if self.n == 0 {
            self.last_arrived_at = arrived_at;
            self.n += 1;
            return;
        }


        if self.n - 1 == self.window_length {
            self.arrival_intervals.remove(0);
            self.n -= 1;
        }
        if self.n != 0 {
            let arrival_interval = arrived_at.sub(self.last_arrived_at).num_milliseconds() as u64;
            self.arrival_intervals.push(arrival_interval);
        }
        self.last_arrived_at = arrived_at;
        self.n += 1;
    }
}

/// PhiCore trait for mean and variance calculation
#[async_trait]
trait PhiCore {
    /// Calculate mean with existing stats.
    async fn mean_with_stats<'a>(&self, stats: Arc<RwLockReadGuard<'a, Statistics>>) -> Result<f64, Box<dyn Error>>;

    /// Calculate variance and mean with existing stats.
    async fn variance_and_mean(&self) -> Result<(f64, f64), Box<dyn Error>>;
}

/// PhiInteraction trait for Detector
#[async_trait]
pub trait PhiInteraction {
    /// Insertion of heartbeat arrival time.
    async fn insert(&self, arrived_at: DateTime<Local>) -> Result<(), Box<dyn Error>>;

    /// Trait for phi for implementing struct
    async fn phi(&self, t: DateTime<Local>) -> Result<f64, Box<dyn Error>>;

    /// Last arrival time of heartbeat
    async fn last_arrived_at(&self) -> Result<DateTime<Local>, Box<dyn Error>>;
}

/// Implementation of PhiCore for Detector
#[async_trait]
impl PhiCore for Detector {
    async fn mean_with_stats<'a>(&self, stats: Arc<RwLockReadGuard<'a, Statistics>>) -> Result<f64, Box<dyn Error>> {
        let mut mean: f64 = 0.;
        let len = &stats.arrival_intervals.len();
        for v in &stats.arrival_intervals {
            mean += *v as f64 / *len as f64;
        }
        Ok(mean)
    }

    async fn variance_and_mean(&self) -> Result<(f64, f64), Box<dyn Error>> {
        let mut variance: f64 = 0.;
        let stats = Arc::new(self.statistics.read().await);
        let mu = self.mean_with_stats(Arc::clone(&stats)).await?;
        let len = &stats.arrival_intervals.len();
        for v in &stats.arrival_intervals {
            let val = ((*v as f64 - mu) * (*v as f64 - mu)) / *len as f64;
            variance += val;
        }
        Ok((variance, mu))
    }
}

/// Cumulative distribution function for normal distribution
fn normal_cdf(t: f64, mu: f64, sigma: f64) -> f64 {
    if sigma == 0. {
        return if t >= mu {
            1.
        } else {
            0.
        };
    }

    let z = (t - mu) / sigma;
    0.5 + 0.5 * erf(z / std::f64::consts::SQRT_2)
}

/// Implementation of PhiInteraction for Detector
#[async_trait]
impl PhiInteraction for Detector {
    async fn insert(&self, arrived_at: DateTime<Local>) -> Result<(), Box<dyn Error>> {
        let mut stats = self.statistics.write().await;
        stats.insert(arrived_at);
        Ok(())
    }

    async fn phi(&self, t: DateTime<Local>) -> Result<f64, Box<dyn Error>> {
        let (sigma_sq, mu) = self.variance_and_mean().await?;
        let sigma = sigma_sq.sqrt();
        let last_arrived_at = self.last_arrived_at().await?;
        let time_diff = t.sub(last_arrived_at).sub(self.acceptable_pause);
        let ft = normal_cdf(time_diff.num_milliseconds() as f64, mu, sigma);
        let phi = -log10(1. - ft);
        Ok(phi)
    }

    async fn last_arrived_at(&self) -> Result<DateTime<Local>, Box<dyn Error>> {
        Ok(self.statistics.read().await.last_arrived_at)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;
    use chrono::{Duration, Local, TimeDelta};
    use tokio::sync::RwLock;
    use crate::{Detector, PhiCore, PhiInteraction, Statistics};

    #[tokio::test]
    async fn test_variant_mean_and_variance_combo_calculation() {
        let mut stats = Statistics::new(10);
        let mut i = 0;
        let mut curr_time = Local::now();
        &stats.insert(curr_time.clone());
        let expect_vals = [1630, 4421, 1514, 216, 231, 931, 4182, 102, 104, 241, 5132];
        while i < expect_vals.len() {
            curr_time = curr_time.add(Duration::milliseconds(expect_vals[i]));
            let arrived_at = curr_time;
            &stats.insert(arrived_at);
            i += 1;
        }
        let detector = Detector {
            statistics: RwLock::new(stats),
            acceptable_pause: TimeDelta::milliseconds(0),
        };
        let (mut variance, mut mean) = detector.variance_and_mean().await.unwrap();
        mean = (mean * 100.0).round() * 0.01;
        variance = (variance * 100.0).round() * 0.01;
        assert_eq!(1707.4, mean);
        assert_eq!(3755791.64, variance);

        let mut suspicion_level: Vec<f64> = vec![];
        for i in 1..10 {
            curr_time = curr_time.add(Duration::milliseconds(250));
            suspicion_level.push(detector.phi(curr_time).await.unwrap())
        }
        println!("suspicion -> {:?}", suspicion_level);
        for i in 1..suspicion_level.len() {
            assert!(suspicion_level[i] > suspicion_level[i - 1]);
        }
    }

    #[tokio::test]
    async fn test_constant_phi_with_constant_pings_calculation() {
        let stats = Statistics::new(10);
        let detector = Detector {
            statistics: RwLock::new(stats),
            acceptable_pause: TimeDelta::milliseconds(0),
        };
        let mut i = 0;
        let mut curr_time = Local::now();
        while i <= 100 {
            let arrived_at = curr_time;
            &detector.insert(arrived_at).await;
            curr_time = curr_time.add(Duration::milliseconds(10));
            i += 10;
        }
        let (mut variance, mut mean) = detector.variance_and_mean().await.unwrap();
        mean = (mean * 100.0).round() * 0.01;
        variance = (variance * 100.0).round() * 0.01;
        assert_eq!(10., mean);
        assert_eq!(0., variance);
        curr_time = curr_time.add(Duration::milliseconds(10));
        assert_eq!(0., detector.phi(curr_time).await.unwrap());
    }

    /// P_later(t) from Eq. (3) of Hayashibara et al., evaluated by composite Simpson
    /// integration of the normal density, so it does not depend on `erf`.
    fn paper_p_later(t: f64, mu: f64, sigma: f64) -> f64 {
        let density = |x: f64| {
            (-(x - mu) * (x - mu) / (2. * sigma * sigma)).exp() / (sigma * (2. * std::f64::consts::PI).sqrt())
        };
        let upper = mu + 40. * sigma;
        if t >= upper {
            return 0.;
        }
        let steps = 200_000;
        let h = (upper - t) / steps as f64;
        let mut sum = density(t) + density(upper);
        for k in 1..steps {
            let weight = if k % 2 == 1 { 4. } else { 2. };
            sum += weight * density(t + k as f64 * h);
        }
        sum * h / 3.
    }

    async fn detector_with_intervals(intervals: &[i64]) -> (Detector, chrono::DateTime<Local>) {
        let detector = Detector::new(intervals.len() as u32);
        let mut curr_time = Local::now();
        detector.insert(curr_time).await.unwrap();
        for interval in intervals {
            curr_time = curr_time.add(Duration::milliseconds(*interval));
            detector.insert(curr_time).await.unwrap();
        }
        (detector, curr_time)
    }

    #[test]
    fn test_normal_cdf_matches_paper_eq3() {
        let (mu, sigma) = (1000., 100.);
        for i in -30..=30 {
            let z = i as f64 / 10.;
            let t = mu + z * sigma;
            let expected = 1. - paper_p_later(t, mu, sigma);
            let actual = crate::normal_cdf(t, mu, sigma);
            assert!((actual - expected).abs() < 1e-9, "z = {z}: normal_cdf = {actual}, paper F(t) = {expected}");
        }
    }

    #[test]
    fn test_normal_cdf_tabulated_values() {
        // Standard normal table: Phi(1), Phi(2), Phi(-1).
        assert!((crate::normal_cdf(1100., 1000., 100.) - 0.841_344_746_068_543).abs() < 1e-12);
        assert!((crate::normal_cdf(1200., 1000., 100.) - 0.977_249_868_051_821).abs() < 1e-12);
        assert!((crate::normal_cdf(900., 1000., 100.) - 0.158_655_253_931_457).abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_phi_matches_paper_one_sigma_late() {
        // Alternating 900/1100 ms intervals: mean 1000 ms, population std deviation exactly 100 ms.
        let (detector, last) = detector_with_intervals(&[900, 1100, 900, 1100, 900, 1100, 900, 1100, 900, 1100]).await;
        let (variance, mean) = detector.variance_and_mean().await.unwrap();
        assert_eq!((mean, variance), (1000., 10000.));

        let phi = detector.phi(last.add(Duration::milliseconds(1100))).await.unwrap();
        let expected = -paper_p_later(1100., 1000., 100.).log10();
        assert!((expected - 0.799_546).abs() < 1e-5);
        assert!((phi - expected).abs() < 1e-6, "phi = {phi}, paper phi = {expected}");
    }

    #[tokio::test]
    async fn test_phi_accrues_after_constant_heartbeats_stop() {
        // Asymptotic completeness (Property 3): once heartbeats stop, phi must grow without bound.
        let (detector, last) = detector_with_intervals(&[10; 10]).await;
        for silence_ms in [20, 1_000, 3_600_000] {
            let phi = detector.phi(last.add(Duration::milliseconds(silence_ms))).await.unwrap();
            assert_eq!(f64::INFINITY, phi, "phi after {silence_ms} ms of silence was {phi}");
        }
        assert_eq!(0., detector.phi(last.add(Duration::milliseconds(5))).await.unwrap());
    }
}
