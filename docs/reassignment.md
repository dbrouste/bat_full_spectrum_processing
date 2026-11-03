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
