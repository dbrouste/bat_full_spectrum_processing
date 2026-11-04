# Experimental Flattop reassignment (branch Reassignment)

This experimental module does not change `adaptive_v4` or the default public detector.
Unlike the current detection pipeline (Hann STFT), this computes a separate
Flattop STFT and time/frequency reassigned energy.

```python
import soundfile as sf
from bat_analysis.detection import detect_chirps
from bat_analysis.reassignment import (
    reassigned_spectrogram, reassigned_db, refine_detection_peaks,
)

y, sr = sf.read("example.wav")
if y.ndim == 2:
    y = y.mean(axis=1)

r = reassigned_spectrogram(
    y, sr, n_fft=512, hop=64, fmin=20000, fmax=150000,
    noise_percentile=50, snr_min_db=4, snr_full_db=16,
)
# r["reassigned_power"] and r["stft_power"] share the same axes:
# r["times_s"], r["freqs_hz"]
# IMPORTANT: reassigned cells have summed power and are not spectrally
# calibrated against STFT cell values.
v4 = detect_chirps(y, sr)   # unchanged baseline
hybrid = refine_detection_peaks(
    y, sr, v4, n_fft=512, hop=64, snr_min_db=4, snr_full_db=16
)
```

Theory: STFT with g, centered t*g and discrete derivative g'.
Reassigned positions are calculated from complex STFT ratios. Nearest-neighbour
power accumulation makes thin ridges but can fragment weak calls. The progressive
4–16 dB mask weights original bin energy before relocation; it is NOT applied
to the reassigned image. Out-of-band estimates are discarded.

Caveats: 1) reassignment cannot exceed the waveform's information bandwidth;
2) relative and absolute energy calibration differ across windows;
3) zero-padded edges are not analyzed; 4) `noise_percentile` depends heavily
on call occupancy; 5) conservative hybrid mode only adjusts PEAK positions,
not onset/offset, blob topology or ridge modelling.

Next benchmark against manual annotation: compare recall, precision,
frequency/time errors and runtime for Hann adaptive_v4 vs Flattop reassignment
512/1024; specifically inspect weak calls and crossing chirps before
integrating reassigned blobs into candidate generation.

## Experimental trajectory reconstruction

`process_full_spectrum` now accepts `tracking_method="reassignment_flattop"`.
The default remains `"legacy"`. The new mode replaces the **trajectory
tracking stage**, not the SNR blob detector. It follows reassigned Flattop
power in both directions from the supplied seed using a continuity-penalized
local peak selector with a short gap allowance. It does not yet use the legacy
Gaussian line-fit iterations, so these two reconstruction methods must be
benchmarked independently. The result is a three-column array
`[time_seconds, frequency_hz, normalized_amplitude]` in the same 15-ms
chunk-relative temporal convention as the legacy model.

```python
from bat_analysis.modelling import process_full_spectrum

diagnostics = {}
curve = process_full_spectrum(
    y, sr,
    time_mid=candidate["time_mid"],
    duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    broad_fm_seed_freq_hz=candidate.get("broad_fm_seed_freq_hz"),
    tracking_method="reassignment_flattop",
    reassignment_n_fft=512,
    reassignment_hop=64,
    diagnostics=diagnostics,
)
```

Limitations: when the seed is incorrect or separated harmonics overlap, the
greedy path can jump or stop. Reassigned power is sparse and is not a Gaussian
amplitude measure; curve amplitudes are normalized. No claim of improved
accuracy is made before testing against manually annotated WAV files.

## Global trajectory tracker (second iteration)

The former greedy ridge follower is replaced by a second-order dynamic
program with two states: frequency bin and frequency change per frame.
Negative/zero frequency slope (forward in time) is enforced, and changes in
slope are penalized. The evidence map blends smoothed reassigned power (65%)
and regular Flattop STFT power (35%) after per-frequency noise subtraction.
This reduces jumps between disconnected reassigned pixels.

The optional entry point remains:

```python
curve = process_full_spectrum(
    y, sr, time_mid=candidate["time_mid"], duration=candidate["duration"],
    seed_freq_hz=candidate["peak_freq_hz"],
    tracking_method="reassignment_flattop",
    reassignment_n_fft=512, reassignment_hop=48,
)
```

**Important:** the preceding example uses the detector's seed. The available
manual-label benchmark is *oracle seeded*, meaning seed time and seed frequency
are computed from manual chirp annotations, not from the detector. It therefore
evaluates trajectory reconstruction separately from detector reliability.

For a repeatable annotated benchmark:

```bash
python -m benchmark.reassignment_ridges Nycto-Thomas.json /path/to/wavs
```

Development run (140 manually annotated calls, four WAVs, 384 kHz),
computed on the prototype prior to GitHub integration:

| WAV | annotated | trajectories returned | median absolute frequency error |
|---|---:|---:|---:|
| 20261005_191936.wav | 61 | 57 | 0.92 kHz |
| 20261005_191947.wav | 38 | 26 | 1.47 kHz |
| 20261005_192441.wav | 27 | 24 | 1.05 kHz |
| 20261005_192501.wav | 14 | 14 | 0.80 kHz |

Errors measured only over extracted points inside the annotated temporal
bounds against PCHIP annotations (linear interpolation for two-point labels).
The failures are disproportionately short/high-frequency harmonic traces.
These results are not a benchmark of autonomous seed placement, nor a
side-by-side comparison against the legacy tracker.
