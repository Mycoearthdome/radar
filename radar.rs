use std::collections::HashMap;
use std::f64::consts::PI;

/// Constants
const SPEED_OF_LIGHT: f64 = 3.0e8; // Speed of light in m/s
const BOLTZMANN_CONSTANT: f64 = 1.38e-23; // Boltzmann constant in J/K
const EARTH_RADIUS: f64 = 6.371e6; // Earth radius in meters

/// Environmental Factors
#[derive(Debug, Clone)]
struct Environment {
    temperature: f64,        // Temperature in Kelvin
    humidity: f64,           // Relative humidity in percentage
    pressure: f64,           // Atmospheric pressure in Pascals
    clutter_type: String,    // Clutter type: "urban", "sea", "rural"
    noise_figure: f64,       // Noise figure of the receiver in dB
    attenuation_factor: f64, // Atmospheric attenuation factor (specific to environment)
}

/// Radar Parameters
#[derive(Debug, Clone)]
struct Radar {
    frequency: f64,           // Operating frequency in Hz
    power: f64,               // Transmitted power in Watts
    gain: f64,                // Antenna gain in linear scale
    pulse_width: f64,         // Pulse width in seconds
    prf: f64,                 // Pulse repetition frequency in Hz
    range_resolution: f64,    // Range resolution in meters
    beam_width: f64,          // Beam width in degrees
    detection_threshold: f64, // Detection threshold in dB
}

/// Target Properties
#[derive(Debug, Clone)]
struct Target {
    range: f64,          // Distance to the target in meters
    rcs: f64,            // Radar cross-section in square meters
    velocity: f64,       // Radial velocity in m/s
    aspect_angle: f64,   // Aspect angle in degrees
    is_hypersonic: bool, // Flag to indicate if target is hypersonic
}

/// Signal Processing
struct Signal {
    samples: Vec<f64>,  // Sampled signal
    doppler_shift: f64, // Doppler shift in Hz
    snr: f64,           // Signal-to-noise ratio
}

impl Signal {
    fn new(samples: usize) -> Self {
        Signal {
            samples: vec![0.0; samples],
            doppler_shift: 0.0,
            snr: 0.0,
        }
    }
}

/// Radar Framework
struct RadarFramework {
    radar: Radar,
    environment: Environment,
    targets: Vec<Target>,
}

impl RadarFramework {
    fn new(radar: Radar, environment: Environment) -> Self {
        RadarFramework {
            radar,
            environment,
            targets: Vec::new(),
        }
    }

    /// Add a target to the simulation
    fn add_target(&mut self, target: Target) {
        self.targets.push(target);
    }

    /// Compute the radar range equation for a target
    fn radar_range_equation(&self, target: &Target) -> f64 {
        let wavelength = SPEED_OF_LIGHT / self.radar.frequency;
        let numerator =
            self.radar.power * self.radar.gain.powi(2) * wavelength.powi(2) * target.rcs;
        let denominator = (4.0 * PI).powi(3) * target.range.powi(4);
        numerator / denominator
    }

    /// Compute Doppler shift for a target
    fn compute_doppler_shift(&self, target: &Target) -> f64 {
        2.0 * target.velocity * self.radar.frequency / SPEED_OF_LIGHT
    }

    /// Simulate signal reflection and reception
    fn simulate(&self) -> Signal {
        let samples = 1024; // Number of samples
        let mut signal = Signal::new(samples);

        for target in &self.targets {
            let received_power = self.radar_range_equation(target);

            // Doppler shift
            let doppler_shift = self.compute_doppler_shift(target);
            signal.doppler_shift += doppler_shift;

            // Add noise and environmental attenuation
            let noise_power = self.compute_noise_power();
            let attenuation = self.compute_attenuation(target.range);
            let total_power = received_power * attenuation - noise_power;

            // SNR calculation
            signal.snr = 10.0 * (total_power / noise_power).log10();

            // Adjust signal for hypersonic effects
            if target.is_hypersonic {
                let plasma_attenuation = self.compute_plasma_attenuation(target.velocity);
                signal.snr *= plasma_attenuation;
            }

            // Fill the signal samples (simplified model)
            for sample in signal.samples.iter_mut() {
                *sample += total_power;
            }
        }
        signal
    }

    fn fine_tune_radar(&mut self) -> (String, f64) {
        let mut results = HashMap::new();
        let mut frequency_range = vec![];
        // Radar tuning ranges
        let mut frequency = 8e9;
        while frequency <= 12e9 {
            frequency_range.push(frequency);
            frequency += 1e9; //GHz
        }
        let power_range = (500..=1500).step_by(250).map(|p| p as f64); // Watts
        let gain_range = (20..=40).step_by(5).map(|g| g as f64); // dB
        let mut pulse_width_range = vec![];
        let mut pulse_width = 1e-7;
        while pulse_width <= 1e-5 {
            pulse_width_range.push(pulse_width);
            pulse_width += 1e-6;
        }
        let prf_range = (500..=2000).step_by(500).map(|prf| prf as f64); // Hz
        let detection_threshold_range = (-20..=0).step_by(5).map(|dt| dt as f64); // dB

        // Environmental conditions
        let temperature_range = (270..=310).step_by(10).map(|t| t as f64); // Kelvin
        let humidity_range = (10..=90).step_by(20).map(|h| h as f64); // Percentage
        let clutter_types = vec!["urban", "sea", "rural"];
        let attenuation_factors = vec![0.00005, 0.0001, 0.0002]; // Attenuation constants

        // Targets
        let max_range = 5000.0; // Max range in meters
        let target_step = 100.0; // Step size for target ranges
        let num_steps = (max_range / target_step) as usize;

        for frequency in frequency_range {
            for power in power_range.clone() {
                for gain in gain_range.clone() {
                    for pulse_width in pulse_width_range.clone() {
                        for prf in prf_range.clone() {
                            for detection_threshold in detection_threshold_range.clone() {
                                // Update radar configuration
                                self.radar.frequency = frequency;
                                self.radar.power = power;
                                self.radar.gain = gain;
                                self.radar.pulse_width = pulse_width;
                                self.radar.prf = prf;
                                self.radar.detection_threshold = detection_threshold;

                                let mut total_launched = 0;
                                let mut total_detected = 0;

                                for temperature in temperature_range.clone() {
                                    for humidity in humidity_range.clone() {
                                        for clutter_type in &clutter_types {
                                            for attenuation_factor in &attenuation_factors {
                                                // Update environment
                                                self.environment.temperature = temperature;
                                                self.environment.humidity = humidity;
                                                self.environment.clutter_type =
                                                    clutter_type.to_string();
                                                self.environment.attenuation_factor =
                                                    *attenuation_factor;

                                                for step in 0..=num_steps {
                                                    let range = step as f64 * target_step;
                                                    let target = Target {
                                                        range,
                                                        rcs: 1.0 + (range / max_range), // Vary RCS linearly
                                                        velocity: if step % 2 == 0 {
                                                            300.0
                                                        } else {
                                                            1500.0
                                                        }, // Conventional and hypersonic
                                                        aspect_angle: 45.0, // Default aspect angle
                                                        is_hypersonic: step % 2 != 0,
                                                    };
                                                    total_launched += 1;

                                                    // Add target and simulate
                                                    self.add_target(target);
                                                    let signal = self.simulate();

                                                    // Check detection based on SNR and threshold
                                                    if signal.snr > self.radar.detection_threshold {
                                                        total_detected += 1;
                                                    }

                                                    // Remove target after simulation
                                                    self.targets.pop();
                                                }
                                            }
                                        }
                                    }
                                }

                                // Compute detection rate
                                let detection_rate = if total_launched > 0 {
                                    total_detected as f64 / total_launched as f64
                                } else {
                                    0.0
                                };

                                // Save result in the hashmap
                                let config_key = format!(
                                    "Freq: {:.1}GHz, Power: {:.1}W, Gain: {:.1}dB, Pulse Width: {:.1e}s, PRF: {:.1}Hz, Threshold: {:.1}dB",
                                    frequency / 1e9, power, gain, pulse_width, prf, detection_threshold
                                );
                                results.insert(config_key, detection_rate);
                            }
                        }
                    }
                }
            }
        }

        // Find the best configuration
        let best_config = results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        (best_config.0.clone(), *best_config.1)
    }

    /// Compute noise power
    fn compute_noise_power(&self) -> f64 {
        let bandwidth = 1.0 / self.radar.pulse_width; // Bandwidth in Hz
        let noise_figure_linear = 10f64.powf(self.environment.noise_figure / 10.0);
        BOLTZMANN_CONSTANT * self.environment.temperature * bandwidth * noise_figure_linear
    }

    /// Compute atmospheric attenuation
    fn compute_attenuation(&self, range: f64) -> f64 {
        (-self.environment.attenuation_factor * range).exp()
    }

    /// Compute plasma attenuation for hypersonic objects
    fn compute_plasma_attenuation(&self, velocity: f64) -> f64 {
        let ionization_factor = 1e-6; // Arbitrary constant for demonstration
        (-ionization_factor * velocity).exp()
    }
}

fn main() {
    // Define radar parameters
    let radar = Radar {
        frequency: 10e9, // Initial values, to be tuned
        power: 1e3,
        gain: 30.0,
        pulse_width: 1e-6,
        prf: 1e3,
        range_resolution: 15.0,
        beam_width: 3.0,
        detection_threshold: -10.0,
    };

    // Define environment
    let environment = Environment {
        temperature: 290.0, // Defaults, to be varied
        humidity: 50.0,
        pressure: 101325.0,
        clutter_type: "urban".to_string(),
        noise_figure: 3.0,
        attenuation_factor: 0.0001,
    };
    // Define parameter ranges for exploration
    let frequencies = (8..=12).map(|f| f as f64 * 1e9); // 8 GHz to 12 GHz
    let powers = (500..=1500).step_by(250).map(|p| p as f64); // 500W to 1500W
    let gains = (20..=40).step_by(5).map(|g| g as f64); // 20 dB to 40 dB
    let pulse_widths = (0..=10).map(|i| 1e-7 + i as f64 * 1e-6); // 1e-7 to 1e-5 seconds
    let prfs = (500..=2000).step_by(500).map(|prf| prf as f64); // 500 Hz to 2000 Hz

    // Simulation parameters
    let target_rcs = 1.0; // Target radar cross-section (m^2)
    let target_range = 5000.0; // Target range (m)

    // Optimization variables
    let mut best_config = radar.clone();
    let mut best_detection_rate = f64::MIN;

    let mut configuration_count = 1;
    // Iterate over all possible configurations
    for frequency in frequencies {
        for power in powers.clone() {
            for gain in gains.clone() {
                for pulse_width in pulse_widths.clone() {
                    for prf in prfs.clone() {
                        // Create a radar configuration
                        let radar = Radar {
                            frequency,
                            power,
                            gain,
                            pulse_width,
                            prf,
                            range_resolution: radar.range_resolution,
                            beam_width: radar.beam_width,
                            detection_threshold: radar.detection_threshold,
                        };

                        // Create radar framework
                        let mut framework = RadarFramework::new(radar, environment.clone());

                        // Fine-tune radar and retrieve best configuration
                        let (best_config, best_detection_rate) = framework.fine_tune_radar();

                        println!("Radar Configuration#{}:", configuration_count);
                        println!("Frequency: {:.1} GHz", framework.radar.frequency / 1e9);
                        println!("Power: {:.1} W", framework.radar.power);
                        println!("Gain: {:.1} dB", framework.radar.gain);
                        println!("Pulse Width: {:.1e} s", framework.radar.pulse_width);
                        println!("PRF: {:.1} Hz", framework.radar.prf);
                        println!("Range Resolution: {:.1} m", framework.radar.range_resolution);
                        println!("Beam Width: {:.1}°", framework.radar.beam_width);
                        println!("Detection Threshold: {:.1} dB", framework.radar.detection_threshold);
                        println!("----------------------------------------------------------------------");
                        println!("Best Radar Configuration: {}", best_config);
                        println!("Best Detection Rate: {:.2}%", best_detection_rate * 100.0);
                        println!("----------------------------------------------------------------------");
                        configuration_count += 1;
                    }
                }
            }
        }
    }
}
