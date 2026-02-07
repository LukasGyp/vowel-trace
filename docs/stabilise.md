---

````markdown
# IMPLEMENTATION_PLAN.md
## Real-time Formant Tracking Stabilization
### Kalman Filter & Viterbi Algorithm Integration

---

## 1. Background and Problem Statement

The current Rust (cargo) project performs **real-time formant estimation** from audio input and displays the results in a GUI.
The existing estimator works reliably for **sustained monophonic vowels**, but becomes unstable in continuous speech, especially when **consonants** are present.

Observed failure modes include:
- Sudden jumps of estimated formant frequencies (particularly F1 and F2)
- Temporary loss of estimates during unvoiced or noisy segments
- Visually unstable trajectories in the GUI
- Lack of temporal consistency between consecutive frames

The root cause is that the current pipeline treats each frame almost independently and does not explicitly model **temporal continuity**.

---

## 2. Objective

The goal is to improve robustness and temporal stability of formant trajectories by introducing **time-aware tracking methods**, while keeping the existing estimator as the measurement source.

Two complementary solutions shall be implemented:

1. **Kalman Filter–based tracking**
   - Continuous-state, smooth tracking
   - Robust to short dropouts
   - Low latency

2. **Viterbi Algorithm–based tracking**
   - Discrete-state sequence optimization
   - Explicit transition constraints
   - Robust against spurious peaks during consonants

Both solutions must be:
- Selectable at runtime via the GUI
- Integrated without breaking the existing audio/GUI architecture
- Designed so that additional tracking methods can be added later

---

## 3. High-Level Design Principles

### 3.1 Separation of Concerns

The processing pipeline shall be conceptually split into:

- **Measurement**  
  Frame-wise formant estimation (existing code).  
  Output may be noisy or missing.

- **Tracking**  
  Temporal integration of measurements into stable trajectories.

This separation allows different tracking strategies to reuse the same measurement backend.

---

### 3.2 Common Data Model

All trackers operate on the same formant representation.

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct Formants {
    pub f1_hz: f32,
    pub f2_hz: f32,
    pub f3_hz: f32,
}
````

Measurements may be missing for a given frame:

```rust
Option<Formants>
```

If the existing estimator provides a quality metric (e.g. peak sharpness, LPC residual),
it may later be extended to:

```rust
pub struct MeasuredFormants {
    pub formants: Formants,
    pub confidence: f32, // 0.0 – 1.0
}
```

This extension is optional and not required for the first implementation.

---

## 4. Tracker Abstraction

All tracking methods shall implement a common interface.

### 4.1 Tracker Trait

```rust
pub trait Tracker {
    fn reset(&mut self, sample_rate: f32);
    fn update(
        &mut self,
        dt: f32,
        measurement: Option<Formants>
    ) -> Option<Formants>;
}
```

* `dt` is the time step between frames
* Missing measurements (`None`) must be handled gracefully
* Returned value is the best estimate for display

---

### 4.2 Tracking Modes

```rust
pub enum TrackingMode {
    Raw,
    Kalman,
    Viterbi,
}
```

* `Raw`: bypass tracking and use the measurement directly
* `Kalman`: continuous-state Kalman filter
* `Viterbi`: discrete-state Viterbi tracking

The active mode must be changeable at runtime via the GUI.

---

## 5. Module Structure

The following module layout is recommended (adapt if the existing structure differs):

```
src/
  tracking/
    mod.rs
    tracker.rs      // Tracker trait and enums
    kalman.rs       // KalmanTracker implementation
    viterbi.rs      // ViterbiTracker implementation
  dsp/
    measurement.rs  // Existing or refactored formant estimation
  ui/
    controls.rs     // GUI controls for mode & parameters
```

---

## 6. Solution A: Kalman Filter Tracking

### 6.1 State Model

A constant-velocity model is used.

State vector:

```
x = [F1, F1_dot, F2, F2_dot, F3, F3_dot]^T
```

Observation vector:

```
z = [F1, F2, F3]^T
```

* Transition: linear constant-velocity
* Observation: direct position measurement
* Missing measurements trigger prediction-only updates

---

### 6.2 Noise Models

* Process noise `Q`: controls how freely formants can move
* Measurement noise `R`: reflects estimator uncertainty
* Initial implementation may use diagonal matrices only

Optional gating:

* If `|measurement - prediction| > max_jump_hz`, ignore that component

---

### 6.3 Kalman Acceptance Criteria

* Stable vowel trajectories with minimal added latency
* No divergence or NaN values during dropouts
* Reduced spikes compared to raw estimation

---

## 7. Solution B: Viterbi Algorithm Tracking

### 7.1 Approach (Initial Version)

A **grid-based Hidden Markov Model** is used for F1 and F2 only.

* State: `(F1_bin, F2_bin)`
* Frequency ranges:

  * F1: 150–1200 Hz
  * F2: 500–4000 Hz
* Bin width: 25–50 Hz

F3 remains taken from the raw estimator in this first version.

---

### 7.2 Cost Functions

* **Emission cost**: distance between grid state and raw measurement
* **Transition cost**: penalizes large inter-frame jumps
* **Dropout cost**: allows missing measurements without collapse

---

### 7.3 Online Operation

For real-time use, a fixed-size sliding window is applied:

* Keep last `T` frames (e.g. 20–40)
* Re-run Viterbi on the window each update
* Output the most recent state estimate

---

### 7.4 Viterbi Acceptance Criteria

* Fewer catastrophic jumps during consonants
* Plausible trajectories even with incorrect raw peaks
* Acceptable CPU load for real-time GUI display

---

## 8. GUI Integration

### 8.1 Required Controls

* Tracking mode selector (`Raw / Kalman / Viterbi`)
* Kalman parameters:

  * Process noise
  * Measurement noise
  * Max jump threshold
* Viterbi parameters:

  * Bin width
  * Transition smoothness
  * Dropout penalty

Changes must take effect immediately without restarting audio capture.

---

### 8.2 Thread Safety

* Audio processing thread owns tracker state
* GUI thread only updates shared parameters
* Use `Arc<Mutex<_>>` or atomics as appropriate

---

## 9. Implementation Steps (for Codex)

1. Analyze the existing codebase to identify:

   * Frame processing loop
   * Measurement generation
   * GUI data flow
2. Introduce the `tracking` module and `Tracker` trait.
3. Refactor the pipeline to route measurements through a selected tracker.
4. Implement `KalmanTracker` with missing-data handling.
5. Implement `ViterbiTracker` (grid-based F1/F2, sliding window).
6. Add GUI controls for mode selection and parameters.
7. Ensure the project builds and runs (`cargo build`, `cargo fmt`).

---

## 10. Manual Test Scenarios

* Sustained vowel: smooth, stable trajectories
* Continuous speech with plosives/fricatives: reduced spikes
* Silence/unvoiced segments: no numerical instability
* Runtime mode switching: no crashes or deadlocks

---

## 11. Notes and Future Extensions

* Add confidence-weighted Kalman updates
* Extend Viterbi to candidate-based tracking instead of grids
* Include F3 in the Viterbi state
* Persist parameter presets per tracking mode

---

**This document is the authoritative implementation plan.
Codex should treat it as a specification and modify the codebase accordingly.**

```
