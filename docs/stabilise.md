---

````markdown
# IMPLEMENTATION_PLAN.md
## Real-time Formant Tracking Stabilization (Revised for Current Code)
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
Another major cause is **mis-assignment of formant identities** when a lower formant is missing (e.g., F1 not detected but F2 exists),
which currently leads to F2 being incorrectly used as F1.

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

### 3.2 Common Data Model (Aligned to Current Code)

The current pipeline tracks **F1 and F2 only** and uses `f64` in the primary data flow:

```rust
#[derive(Debug, Clone)]
pub(crate) struct FormantPoint {
    pub(crate) t: f64,
    pub(crate) f1: f64,
    pub(crate) f2: f64,
}
````

Tracking must integrate with this existing `FormantPoint` stream to avoid broad refactors.

To enable candidate-based tracking while remaining compatible, introduce a **parallel measurement type**
and keep `FormantPoint` as the output:

```rust
#[derive(Clone, Copy, Debug)]
pub struct FormantCandidate {
    pub freq_hz: f64,
    pub bandwidth_hz: f64,
    pub strength: f32, // 0.0 – 1.0
}

#[derive(Clone, Debug)]
pub struct FormantCandidates {
    pub candidates: Vec<FormantCandidate>,
    pub voiced: bool,
    pub voicing_conf: f32, // 0.0 – 1.0
}
```

Notes based on existing DSP:
- LPC formant candidates already exist implicitly as **roots filtered by frequency/bandwidth**.
- `freq_hz` is computed from pole angle; `bandwidth_hz` from pole radius.
- `strength` should be derived from pole radius `r` with a monotonic map, e.g.:
  `strength = clamp01((r - r_min) / (r_max - r_min))`.  
  Default `r_min/r_max` must be documented (see Section 6.2).

**Voicing confidence** currently only exists as an RMS gate:
- `voiced = rms >= rms_gate`
- `rms_gate` is adaptive and already computed in `processing.rs`.

If additional voicing features (ZCR/AC) are desired later, they must be **explicitly added**,
but the first implementation must remain compatible with the current RMS gate to avoid breaking behavior.

---

## 4. Tracker Abstraction

All tracking methods shall implement a common interface.

### 4.1 Tracker Trait

```rust
pub trait Tracker {
    fn reset(&mut self, sample_rate_hz: f32);
    fn update(
        &mut self,
        dt: f32,
        measurement: Option<FormantCandidates>,
    ) -> Option<FormantPoint>;
}
```

* `dt` is the time step between frames
* Missing measurements (`None`) must be handled gracefully
* Returned value is the best estimate for display and must match the existing GUI data flow.
* Trackers must use `voicing_conf` to down-weight unreliable frames.
* `None` means "no candidates available"; low-confidence frames should still pass as `Some(...)`
  with low `voicing_conf`.

**`dt` handling rule (required):**
- `dt` is fixed to the hop size (`HOP_SEC = 0.010`) in the current pipeline.
- Trackers may cache `dt` internally on `reset()` for efficiency, but `update()` must still accept it.

**Output rule for missing data (required):**
- If the tracker cannot produce F1/F2 for a frame, it must return `None`.
- The GUI currently interprets gaps by time (`gap = 0.3s`). Returning `None` will naturally create gaps
  because no `FormantPoint` is pushed for that frame.
- Do not emit `NaN` or sentinel frequencies into `FormantPoint`.

---

### 4.2 Tracking Modes

```rust
pub enum TrackingMode {
    Raw,
    Kalman,
    Viterbi,
}
```

* `Raw`: bypass tracking and use the measurement directly (assign from candidates)
* `Kalman`: continuous-state Kalman filter with candidate-based measurement assignment
* `Viterbi`: candidate-based sequence optimization (no grid quantization)

The active mode must be changeable at runtime via the GUI.

**Raw mode assignment rule (required):**
- From candidates within the F1/F2 ranges, select up to two by **highest strength**.
- If two are selected, assign the lower frequency to F1 and the higher to F2.
- If only one candidate exists, output `None` (do not guess F2 or reuse prior values).

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
    measurement.rs  // Existing or refactored formant estimation + candidate extraction
  ui/
    controls.rs     // GUI controls for mode & parameters
```

**Current code has no `dsp/` or `tracking/` directories.**
Initial integration should minimize churn:
- Keep `processing.rs` as the owner of candidate extraction.
- Add `tracking/` module and call it from `spawn_processing_thread`.
- Extend `types.rs` with `FormantCandidates` and `FormantCandidate`.

---

## 6. Solution A: Kalman Filter Tracking

### 6.1 State Model

A constant-velocity model is used.

State vector (F1/F2 only to match current GUI):

```
x = [F1, F1_dot, F2, F2_dot]^T
```

Observation vector:

```
z = [F1, F2]^T
```

* Transition: linear constant-velocity
* Observation: direct position measurement
* Missing measurements trigger prediction-only updates
* Candidate-based data association is required:
  - Choose the most plausible candidate for each formant using prediction proximity,
    strength, and the constraint `F1 < F2`.
  - If no plausible candidate exists, treat that component as missing.

**Silence reset rule (required):**
- If `voicing_conf < 0.2` for `N_silence = 30` consecutive frames (~0.3s at 10ms hop),
  call `reset()` on the tracker state and output `None` until voiced again.

**Required scoring rule (initial default):**
- For each candidate `c` and predicted frequency `f_pred`, define:
  `score(c) = w_d * |c.freq_hz - f_pred| + w_s * (1.0 - c.strength)`
- Default weights: `w_d = 1.0`, `w_s = 200.0` (Hz-equivalent penalty).
- Enforce `F1 < F2` by discarding pairs that violate the order.

**Ordering safeguard (required):**
- If a Kalman update would produce `F1 > F2`, skip that update and keep the prior prediction
  for this frame.

---

### 6.2 Noise Models (Aligned to Current DSP)

* Process noise `Q`: controls how freely formants can move
* Measurement noise `R`: reflects estimator uncertainty
* Initial implementation may use diagonal matrices only

Optional gating:

* If `|measurement - prediction| > max_jump_hz`, down-weight instead of hard reject
  (soft gating) to avoid slow recovery after transitions.

Dynamic noise tuning (required):
* Use candidate strength to scale `R` (low strength -> large `R`).
* When a rapid transition is detected (e.g., large residuals for several frames),
  temporarily increase `Q` to improve tracking speed.

**Clarification for implementation:**  
Define explicit defaults for `r_min/r_max` when computing strength from pole radius.
Recommended initial defaults for the current LPC root range:
- `r_min = 0.7`, `r_max = 0.99` (clamped to [0,1]).
These may be tuned later but must be explicit to keep results deterministic.

**Dynamic Q/R defaults (required):**
- Let residual `e = |z - z_pred|`.
- If `e > 350 Hz` for **3 consecutive frames**, multiply `Q` by `4.0` for the next **5 frames**.
- `R` is scaled by `(1.0 - strength)^2`, clamped to `[0.5, 4.0]` multiplier.

---

### 6.3 Kalman Acceptance Criteria

* Stable vowel trajectories with minimal added latency
* No divergence or NaN values during dropouts
* Reduced spikes compared to raw estimation
* Faster recovery at vowel-consonant transitions

---

## 7. Solution B: Viterbi Algorithm Tracking

### 7.1 Approach (Initial Version)

A **candidate-based Hidden Markov Model** is used for F1 and F2 only.

* State: `(F1_candidate, F2_candidate)`
* Candidates are continuous-valued; no grid quantization.
* Frequency ranges are enforced by candidate filtering:
  * F1: 150–1200 Hz
  * F2: 500–4000 Hz

F3 is not used in the current application (GUI and pipeline are F1/F2 only).

---

### 7.2 Cost Functions

* **Emission cost**:
  - Preference for higher `strength`
  - Penalize implausible bandwidths
  - Penalize F1/F2 ordering violations
* **Transition cost**: continuous distance penalty between candidate frequencies
* **Dropout cost**: allows missing candidates without collapse

**Default Viterbi costs (required):**
- Emission cost:
  `E = w_s * (1.0 - strength) + w_bw * bw_penalty`
  where `bw_penalty = 0.0` if `bandwidth_hz <= 400` else `(bandwidth_hz - 400) / 400`.
- Transition cost:
  `T = w_t * (|f1_t - f1_{t-1}| + |f2_t - f2_{t-1}|)`
- Dropout cost:
  `D = w_d` when no valid candidate pair exists in a frame.

Default weights: `w_s = 2.0`, `w_bw = 1.0`, `w_t = 0.003`, `w_d = 3.0`.

---

### 7.3 Online Operation (Real-time Constraint)

For real-time use, a fixed-size sliding window is applied:

* Keep last `T` frames (e.g. 20–40)
* Re-run Viterbi on the window each update
* Output the most recent state estimate

**Silence reset rule (required):**
- If `voicing_conf < 0.2` for `N_silence = 30` consecutive frames (~0.3s at 10ms hop),
  clear the Viterbi window and output `None` until voiced again.

**Required constraint for real-time use:**  
Limit candidate count per frame (e.g., top 6–8 by strength within range)
to keep per-update runtime bounded. Without this, `O(N^2)` state growth
can become too expensive for the GUI thread.

**Default candidate cap (required):**
- Keep at most `N_F1 = 6` candidates in the F1 range and `N_F2 = 6` in the F2 range
  per frame for Viterbi.

**Candidate pruning (required):**
- Discard candidates with `bandwidth_hz > 800`.
- Discard candidates with `strength < 0.2`.

**Complexity note (informational):**
- With `N_F1 = N_F2 = 6`, the per-frame state count is 36 and transitions are 1296.
- For `T <= 40`, this is acceptable on modern CPUs. Forward-pass reuse is optional.

---

### 7.4 Viterbi Acceptance Criteria

* Fewer catastrophic jumps during consonants
* Plausible trajectories even with incorrect raw peaks
* Acceptable CPU load for real-time GUI display
* Continuous-valued outputs (no grid quantization artifacts)

---

## 8. GUI Integration

### 8.1 Required Controls

* Tracking mode selector (`Raw / Kalman / Viterbi`)
* Kalman parameters:

  * Process noise
  * Measurement noise
  * Max jump threshold
* Viterbi parameters:

  * Transition smoothness
  * Dropout penalty
  * Candidate strength weighting
* Voicing parameters (optional future extension):
  * ZCR normalization range
  * Autocorrelation lag range (pitch range)
  * Power normalization range

**Current implementation must keep RMS gate as the voicing decision.**

**Default GUI values (required):**
- Tracking mode: `Kalman`
- Kalman `Q` (process noise): `2500.0`
- Kalman `R` (measurement noise): `2000.0`
- Kalman `max_jump_hz`: `500.0`
- Viterbi `transition_smoothness` (w_t): `0.003`
- Viterbi `dropout_penalty` (w_d): `3.0`
- Viterbi `strength_weight` (w_s): `2.0`

Changes must take effect immediately without restarting audio capture.

---

### 8.2 Thread Safety

* Audio processing thread owns tracker state
* GUI thread only updates shared parameters
* Use `Arc<Mutex<_>>` or atomics as appropriate

**Note:** To avoid lock contention in the audio thread, prefer atomics or
double-buffered parameter snapshots for frequently updated controls.

**Required synchronization rule:**
- Audio thread must never block on GUI updates.
- GUI updates write to atomics or swap a parameter snapshot once per frame.

---

## 9. Implementation Steps (for Codex)

1. Analyze the existing codebase to identify:
   * Frame processing loop (`spawn_processing_thread` in `processing.rs`)
   * Measurement generation (`lpc_formants_from_coeffs`)
   * GUI data flow (`FormantPoint` stream to `FormantApp`)
2. Introduce the `tracking` module and `Tracker` trait (F1/F2 only).
3. Refactor the pipeline to route candidate measurements through a selected tracker.
4. Implement candidate extraction from LPC roots:
   * `freq_hz` from pole angle
   * `bandwidth_hz` from pole radius
   * `strength` from pole radius (monotonic mapping with explicit `r_min/r_max`)
   * Apply band-limits for candidate lists:
     - F1 candidate list: 150–1200 Hz
     - F2 candidate list: 500–4000 Hz
     - Band-limits are applied **after** generic LPC filtering and before tracking.
5. Implement `KalmanTracker` with candidate-based data association, soft gating,
   and confidence-weighted updates.
6. Implement `ViterbiTracker` (candidate-based F1/F2, sliding window, candidate cap).
7. Add GUI controls for mode selection and parameters.
8. Ensure the project builds and runs (`cargo build`, `cargo fmt`).

---

## 10. Manual Test Scenarios

* Sustained vowel: smooth, stable trajectories
* Continuous speech with plosives/fricatives: reduced spikes
* Silence/unvoiced segments: no numerical instability
* Runtime mode switching: no crashes or deadlocks
* F1-missing frames: F2 should not collapse into F1

**Suggested numeric stability checks (optional):**
- Over a 10s sustained vowel, `P95(|ΔF1|)` and `P95(|ΔF2|)` should be less than 120 Hz.
- Over continuous speech, transient spikes above 500 Hz should be rarer than 2 per second.

---

## 11. Notes and Future Extensions

* Refine confidence mapping from LPC poles (strength calibration)
* Extend Viterbi to joint F1/F2/F3 candidate tracking
* Include F3 in the Viterbi state
* Persist parameter presets per tracking mode

---

**This document is the authoritative implementation plan.
Codex should treat it as a specification and modify the codebase accordingly.**

```
