# Cleaned batch-processing module extracted from a Colab notebook export.
# Keeps the processing pipeline + batch helpers, removes plotting/demo code.

from __future__ import annotations

import os
import math
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import scipy.signal
import scipy.ndimage
import scipy.optimize
from scipy.signal import butter, filtfilt
from scipy.signal import resample
from scipy.signal import resample_poly
from scipy.interpolate import CubicSpline, UnivariateSpline, RectBivariateSpline

# Optional deps (only needed for some metadata/export paths)
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import guano
except Exception:  # pragma: no cover
    guano = None

try:
    import librosa
except Exception as e:  # pragma: no cover
    raise ImportError('This module requires librosa. Install with: pip install librosa') from e


def truncate_audio(y, sr, time_start=None, time_end=None):
    """
    Truncates the audio data `y` based on the specified start and end times.

    Parameters:
    - y: Audio time series (1D NumPy array).
    - sr: Sampling rate of the audio.
    - time_start: Start time in seconds (optional).
    - time_end: End time in seconds (optional).

    Returns:
    - truncated_y: The truncated audio data.
    - new_time_start: The actual start time of the truncated audio.
    - new_time_end: The actual end time of the truncated audio.
    """
    if time_start is None:
        time_start = 0
    if time_end is None:
        time_end = len(y) / sr

    # Convert time to sample indices
    start_sample = max(0, int(time_start * sr))
    end_sample = min(len(y), int(time_end * sr))

    # Truncate the audio data
    truncated_y = y[start_sample:end_sample]
    new_time_start = start_sample / sr
    new_time_end = end_sample / sr

    return truncated_y, new_time_start, new_time_end

## Plot Spectro

def high_pass_filter(y, sr, cutoff=25000, order=8):
    """
    Applies a high-pass filter to an audio signal.

    Parameters:
    - y: Audio time series (1D numpy array).
    - sr: Sampling rate of `y` (in Hz).
    - cutoff: Cutoff frequency for the high-pass filter (in Hz).
    - order: Order of the Butterworth filter (default is 4).

    Returns:
    - Filtered audio signal.
    """
    # Normalize the cutoff frequency to the Nyquist frequency
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist

    # Design the Butterworth high-pass filter
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter to the audio signal
    filtered_signal = filtfilt(b, a, y)

    return filtered_signal

def Median_subtracted_spectrogram(y, sr, reduction=1):
    """
    Process a spectrogram where the median amplitude is subtracted from each frequency bin,
    and the result is converted back to a time-domain signal.

    Parameters:
    - y: Audio time series.
    - sr: Sampling rate of `y`.

    Returns:
    - y_reconstructed: Reconstructed audio signal after median subtraction.
    """

    # Compute the spectrogram in linear amplitude
    stft_result = librosa.stft(y, n_fft=1024, hop_length=512, window='flattop')
    magnitude = np.abs(stft_result)
    phase = np.angle(stft_result)  # Extract phase information

    # Calculate the median amplitude for each frequency bin
    median_amplitudes = np.median(magnitude, axis=1, keepdims=True)

    # Subtract the median amplitude, ensuring non-negative values
    median_subtracted_magnitude = np.maximum(magnitude - reduction * median_amplitudes, 1e-10)

    # Reconstruct the complex spectrogram using the original phase
    modified_stft = median_subtracted_magnitude * np.exp(1j * phase)

    # Convert back to time-domain using ISTFT
    y_reconstructed = librosa.istft(modified_stft, hop_length=512)

    return y_reconstructed

#y, sr = librosa.load('/content/drive/MyDrive/Python/Bat sample/m_rapproche_20250106_213317.wav', sr=None)

## Resample

from scipy.signal import resample_poly

def resample_by_x(signal, original_sr, factor=30):
    upsampled_signal = resample_poly(signal, factor, 1)  # Upsample by factor
    upsampled_sr = original_sr * factor
    return upsampled_signal, upsampled_sr

from scipy.signal import resample
from scipy.signal import resample_poly

def resample_in_chunks(signal, original_sr, factor=30, chunk_size=10000):
    """
    Resample the signal in smaller chunks to avoid memory overflow.

    Parameters:
    - signal (numpy array): Input audio signal.
    - factor (int): Upsampling factor.
    - chunk_size (int): Size of chunks to process at a time.

    Returns:
    - upsampled_signal (numpy array): Resampled signal.
    """
    num_samples = len(signal) * factor
    upsampled_signal = np.zeros(num_samples)  # Preallocate memory

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]
        chunk_resampled = resample_poly(chunk, len(chunk) * factor,1)
        upsampled_signal[i*factor:(i+chunk_size)*factor] = chunk_resampled

    upsampled_sr = original_sr * factor

    return upsampled_signal, upsampled_sr

def resample_by_x3(signal, original_sr, factor=30):
    """
    Resample a signal by 10x.

    Parameters:
    - signal (numpy array): The input audio signal.
    - original_sr (int): The original sampling rate of the signal.

    Returns:
    - upsampled_signal (numpy array): The resampled signal with 10x the original sampling rate.
    - upsampled_sr (int): The new sampling rate after resampling.
    """

    # Define the new sampling rate
    upsampled_sr = original_sr * factor

    # Calculate the number of samples in the resampled signal
    num_samples = len(signal) * factor


    # Resample the signal using scipy's resample function
    upsampled_signal = resample(signal, num_samples)


    return upsampled_signal, upsampled_sr

import numpy as np
from scipy.signal import resample_poly

def compute_zero_crossing_frequency(y, sr, amplitude_threshold=1, chunk_size=10000, factor=30):
    """
    Computes the zero-crossing frequency for each sample in the signal using chunked processing.

    Parameters:
    - y: The input audio signal (1D numpy array).
    - sr: The sampling rate (Hz).
    - amplitude_threshold: Minimum amplitude (0-100) to consider the frequency.
    - chunk_size: Size of chunks for processing.
    - factor: Upsampling factor.

    Returns:
    - zero_crossing_freq: Array of zero-crossing frequencies.
    - zero_crossing_times: Array of times corresponding to zero-crossing events in the full file.
    """
    zero_crossing_freq = []
    zero_crossing_times = []

    for chunk_start in range(0, len(y), chunk_size):
        chunk = y[chunk_start:chunk_start + chunk_size]

        # High-pass filter each chunk
        chunk_highpass = high_pass_filter(chunk, sr, 25000)

        # Upsample chunk
        upsampled_chunk = resample_poly(chunk_highpass, factor, 1)  # More memory-efficient
        upsampled_sr = sr * factor

        # Normalize amplitude
        upsampled_chunk = 100 * (upsampled_chunk / np.max(np.abs(upsampled_chunk)))

        # Find zero crossings
        zero_crossings = np.where(np.diff(np.sign(upsampled_chunk)) == 2)[0]

        # Compute zero-crossing frequency
        for j in range(5, len(zero_crossings), 5):
            start = zero_crossings[j - 5]
            end = zero_crossings[j]
            duration = ((end - start) / upsampled_sr)

            # Calculate segment amplitude
            segment_amplitude = np.max(np.abs(upsampled_chunk[start:end]))

            # Frequency or set to 0 based on amplitude threshold
            if segment_amplitude >= amplitude_threshold:
                zero_crossing_freq.append(5 / duration if duration > 0 else 0)
            else:
                zero_crossing_freq.append(0)

            # Correct time to be relative to the full file
            absolute_time = ((chunk_start * factor) + (start + end) / 2) / upsampled_sr
            zero_crossing_times.append(absolute_time)

    return zero_crossing_freq, zero_crossing_times

def compute_zero_crossing_frequency3(y, sr, amplitude_threshold=1):
    """
    Computes the zero-crossing frequency for each sample in the signal,
    with frequencies set to 0 if the segment amplitude is below a threshold.

    Parameters:
    - y: The input audio signal (1D numpy array).
    - sr: The sampling rate (Hz).
    - amplitude_threshold: Minimum amplitude (0-100) to consider the frequency.

    Returns:
    - zero_crossing_freq: Array of zero-crossing frequencies.
    - zero_crossing_times: Array of times corresponding to zero-crossing events.
    """
    # High-pass filter to remove low-frequency components
    y_highpass = high_pass_filter(y, sr, 25000)
    print_memory_usage()
    #upsample signal
    upsampled_signal, upsampled_sr = resample_by_x(y_highpass, sr, 30)
    print_memory_usage()

    # Normalize the signal amplitude between 0 and 100
    y_normalized = 100 * (upsampled_signal / np.max(np.abs(upsampled_signal)))
    print_memory_usage()
    # Find the zero-crossing points
    zero_crossings = np.where(np.diff(np.sign(y_normalized))== 2)[0]
    print_memory_usage()
    # Store frequency and time every 10 zero crossings
    zero_crossing_freq = []
    zero_crossing_times = []

    for i in range(5, len(zero_crossings), 5):  # Step by 5 zero crossings
        start = zero_crossings[i - 5]
        end = zero_crossings[i]
        duration = ((end - start) / upsampled_sr)

        # Calculate the segment amplitude
        segment_amplitude = np.max(np.abs(y_normalized[start:end]))

        # Compute the frequency or set to 0 based on amplitude threshold
        if segment_amplitude >= amplitude_threshold:
            zero_crossing_freq.append(5 / duration if duration > 0 else 0)
        else:
            zero_crossing_freq.append(0)

        # Record the mid-point time for this zero-crossing segment
        zero_crossing_times.append((start + end) / 2 / upsampled_sr)

    return zero_crossing_freq, zero_crossing_times

# --- SNR/Blob based candidate detection (replacement for ZCR midpoints) ---
def compute_snr_map(
    y: np.ndarray,
    sr: int,
    fmin: float = 20000,
    fmax: float = 150000,
    n_fft: int = 512,
    hop: int | None = None,
    noise_q: float = 20.0,
    noise_mode: str = "percentile",  # "percentile", "lowtrim_mean", "lowtrim_med"
):
    """Compute an SNR map (dB) from a power spectrogram.

    This is meant as a fast candidate detector for bat calls:
    - compute STFT power in dB
    - estimate noise floor per-frequency-bin using either a percentile or a trimmed-low statistic
    - return SNR(dB) = PdB - NdB

    Returns
      snr_db: (F,T) float32
      freqs_b: (F,) Hz (band-limited)
      times: (T,) s
      debug: dict with PdB, NdB (both band-limited)
    """
    y = np.asarray(y, dtype=np.float32)
    if hop is None:
        hop = n_fft // 4

    # scipy.signal.stft is already available via scipy.signal import above
    win = scipy.signal.get_window("hann", n_fft, fftbins=True)
    freqs, times, Z = scipy.signal.stft(
        y,
        fs=sr,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
        return_onesided=True,
    )

    P = (np.abs(Z) ** 2).astype(np.float32)  # power
    PdB = 10.0 * np.log10(P + 1e-20)

    # band-limit
    fmask = (freqs >= fmin) & (freqs <= fmax)
    freqs_b = freqs[fmask]
    PdB = PdB[fmask, :]

    # --- noise floor per frequency bin ---
    T = PdB.shape[1]
    k = max(1, int(np.floor((noise_q / 100.0) * T)))

    if noise_mode == "percentile":
        NdB = np.percentile(PdB, noise_q, axis=1)  # (F,)
    else:
        # take lowest-k values per bin; partition is faster than sort
        low = np.partition(PdB, kth=k - 1, axis=1)[:, :k]  # (F,k)
        if noise_mode == "lowtrim_mean":
            NdB = np.mean(low, axis=1)
        elif noise_mode == "lowtrim_med":
            NdB = np.median(low, axis=1)
        else:
            raise ValueError("noise_mode must be 'percentile', 'lowtrim_mean', or 'lowtrim_med'")

    snr_db = PdB - NdB[:, None]
    debug = {"PdB": PdB, "NdB": NdB}
    return snr_db, freqs_b, times, debug


def _get_filtered_blobs_info(
    mask: np.ndarray,
    times: np.ndarray,
    freqs_b: np.ndarray,
    *,
    min_blob_size: int = 10,
    min_blob_height_hz: float = 5000.0,
    max_blob_slope_hz_per_ms: float = -2000.0,
):
    """Label connected components in a binary mask and apply basic blob filters.

    Notes
    - slope is computed from the lowest and highest frequency pixels in the blob, and
      measured as (f_high - f_low) / (t_at_f_high - t_at_f_low) in Hz/ms.
    - filter keeps blobs whose slope is <= max_blob_slope_hz_per_ms (no abs()) so you can enforce
      'roughly downward chirps' with a negative threshold (e.g. -2000 Hz/ms).
    """
    labeled_mask, num_features = scipy.ndimage.label(mask)
    if num_features == 0:
        return []

    slices = scipy.ndimage.find_objects(labeled_mask)
    filtered = []

    time_res_s = float(times[1] - times[0]) if len(times) > 1 else 1.0
    freq_res_hz = float(freqs_b[1] - freqs_b[0]) if len(freqs_b) > 1 else 1.0

    for i, slc in enumerate(slices):
        blob_region_binary = mask[slc]
        size = int(np.sum(blob_region_binary))

        width_ms = (slc[1].stop - slc[1].start) * time_res_s * 1000.0
        height_hz = (slc[0].stop - slc[0].start) * freq_res_hz

        # slope from min/max frequency pixels within the blob
        slope_hz_per_ms = 0.0
        f_rel, t_rel = np.where(blob_region_binary == 1)
        if f_rel.size > 0:
            f_abs = freqs_b[slc[0].start + f_rel]
            t_abs = times[slc[1].start + t_rel]
            fmin_blob = float(np.min(f_abs))
            fmax_blob = float(np.max(f_abs))
            # representative times at those freqs (mean of all pixels at the extreme freqs)
            t_at_fmin = float(np.mean(t_abs[f_abs == fmin_blob]))
            t_at_fmax = float(np.mean(t_abs[f_abs == fmax_blob]))
            dt_s = t_at_fmax - t_at_fmin
            if dt_s == 0:
                slope_hz_per_ms = np.inf
            else:
                slope_hz_per_ms = ((fmax_blob - fmin_blob) / dt_s) / 1000.0

        pass_filter = True
        if min_blob_size > 0 and size <= min_blob_size:
            pass_filter = False
        if min_blob_height_hz > 0 and height_hz <= min_blob_height_hz:
            pass_filter = False
        if max_blob_slope_hz_per_ms != np.inf and slope_hz_per_ms > max_blob_slope_hz_per_ms:
            pass_filter = False

        if pass_filter:
            # bbox times/freqs (inclusive-ish)
            t0 = float(times[slc[1].start])
            t1 = float(times[slc[1].stop - 1]) if slc[1].stop - 1 < len(times) else float(times[-1])
            f0 = float(freqs_b[slc[0].start])
            f1 = float(freqs_b[slc[0].stop - 1]) if slc[0].stop - 1 < len(freqs_b) else float(freqs_b[-1])

            filtered.append(
                {
                    "slice": slc,
                    "binary_mask_slice": blob_region_binary,
                    "size": size,
                    "width_ms": width_ms,
                    "height_hz": height_hz,
                    "slope_hz_per_ms": slope_hz_per_ms,
                    "t_start": t0,
                    "t_end": t1,
                    "f_low": min(f0, f1),
                    "f_high": max(f0, f1),
                }
            )

    return filtered


def detect_candidates_snr_blobs(
    y: np.ndarray,
    sr: int,
    *,
    snr_threshold_db: float = 10.0,
    percentile_q: float = 96.0,
    fmin: float = 20000,
    fmax: float = 150000,
    n_fft: int = 512,
    hop: int | None = 128,
    min_blob_size: int = 10,
    min_blob_height_hz: float = 5000.0,
    max_blob_slope_hz_per_ms: float = -2000.0,
    echo_suppression_window_ms: float = 10.0,
):
    """Return candidate call centers using an SNR-thresholded blob detector.

    Output is a list of dicts with at least:
      - time_mid (s)
      - duration (s)  (blob bbox duration, min-clamped)
      - peak_freq_hz (Hz)  (freq at max-PdB pixel within the blob)
      - peak_db (dB)
      - t_start, t_end, f_low, f_high, slope_hz_per_ms, size

    Echo suppression:
      keeps only the strongest peak within a +/- window (time) around already-kept peaks.
    """
    snr_map, freqs_b, times, dbg = compute_snr_map(
        y,
        sr,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop=hop,
        noise_q=percentile_q,
        noise_mode="percentile",
    )
    PdB = dbg["PdB"]  # band-limited PdB (F,T)

    mask = (snr_map >= snr_threshold_db).astype(np.uint8)

    blobs = _get_filtered_blobs_info(
        mask,
        times,
        freqs_b,
        min_blob_size=min_blob_size,
        min_blob_height_hz=min_blob_height_hz,
        max_blob_slope_hz_per_ms=max_blob_slope_hz_per_ms,
    )
    if not blobs:
        return []

    all_peaks = []
    for b in blobs:
        slc = b["slice"]
        blob_mask = b["binary_mask_slice"]
        PdB_blob = PdB[slc]
        PdB_masked = np.where(blob_mask == 1, PdB_blob, -np.inf)
        flat = int(np.argmax(PdB_masked))
        f_rel, t_rel = np.unravel_index(flat, PdB_masked.shape)

        f_idx = slc[0].start + f_rel
        t_idx = slc[1].start + t_rel

        peak_freq = float(freqs_b[f_idx])
        peak_time = float(times[t_idx])
        peak_db = float(PdB[f_idx, t_idx])

        # duration from blob bbox (clamp: at least 1 frame)
        t_start = b["t_start"]
        t_end = b["t_end"]
        duration = max(1e-6, t_end - t_start)

        all_peaks.append(
            {
                **b,
                "time_mid": peak_time,
                "duration": duration,
                "peak_freq_hz": peak_freq,
                "peak_db": peak_db,
            }
        )

    # --- Echo suppression (simple time-window NMS) ---
    all_peaks.sort(key=lambda d: d["peak_db"], reverse=True)
    selected = []
    suppressed_intervals = []
    w = echo_suppression_window_ms / 1000.0

    for p in all_peaks:
        t = p["time_mid"]
        if any((s <= t <= e) for (s, e) in suppressed_intervals):
            continue
        selected.append(p)
        suppressed_intervals.append((t - w / 2.0, t + w / 2.0))

    # temporal order
    selected.sort(key=lambda d: d["time_mid"])
    return selected

def detect_stable_frequency_regions(zero_crossing_freq, zero_crossing_times, threshold=2000, min_consecutive=10):
    """
    Identifies positive (stable) and negative (unstable) regions based on zero-crossing frequency stability.

    Parameters:
    - zero_crossing_freq: List or array of zero-crossing frequencies.
    - zero_crossing_times: List or array of times corresponding to zero-crossing events.
    - threshold: Maximum frequency difference (Hz) to be considered stable (default: 2000 Hz).
    - min_consecutive: Minimum number of consecutive points needed for a stable region (default: 10).

    Returns:
    - stable_regions: NumPy array of 1s (stable) and 0s (unstable) corresponding to zero_crossing_times.
    """

    # Convert to numpy array if it's a list
    zero_crossing_freq = np.array(zero_crossing_freq)

    # Initialize an array of zeros (default: unstable)
    stable_regions = np.zeros(len(zero_crossing_freq), dtype=int)

    # Compute pairwise frequency differences
    freq_differences = np.abs(np.diff(zero_crossing_freq))

    # Identify consecutive regions where the frequency difference is below the threshold AND frequency > 0
    stable_indices = []
    for i in range(len(freq_differences) - min_consecutive + 1):
        avg_diff = np.mean(freq_differences[i:i + min_consecutive - 1])

        # Ensure all frequencies in the window are > 0
        if avg_diff < threshold and np.all(zero_crossing_freq[i:i + min_consecutive] > 0):
            stable_indices.append(i)

    # Mark stable regions
    for i in stable_indices:
        stable_regions[i:i + min_consecutive] = 1  # Mark as stable (positive)

    return stable_regions, zero_crossing_times

def compute_stable_region_midpoints(stable_regions, zero_crossing_times):
    """
    Computes the mid-time position and duration of each stable region cluster.

    Parameters:
    - stable_regions: A boolean list/array indicating stability at each zero-crossing.
    - zero_crossing_times: The corresponding times of zero-crossing events.

    Returns:
    - clusters: A list of tuples where each tuple contains (mid_time, duration) for a stable region cluster.
    """
    clusters = []
    in_cluster = False  # Track if we're inside a stable cluster
    cluster_start = None

    for i in range(len(stable_regions)):
        if stable_regions[i]:
            if not in_cluster:  # Start a new cluster
                cluster_start = zero_crossing_times[i]
                in_cluster = True
        else:
            if in_cluster:  # End of a cluster
                cluster_end = zero_crossing_times[i - 1]
                mid_time = (cluster_start + cluster_end) / 2
                duration = cluster_end - cluster_start
                clusters.append((mid_time, duration))
                in_cluster = False

    # Handle case where the last cluster reaches the end of the array
    if in_cluster:
        cluster_end = zero_crossing_times[-1]
        mid_time = (cluster_start + cluster_end) / 2
        duration = cluster_end - cluster_start
        clusters.append((mid_time, duration))

    return clusters


def compute_stable_region_midpoints3(stable_regions, zero_crossing_times):
    """
    Computes the mid-time position and duration of each cluster where stable_region is 1.

    Parameters:
    - stable_regions: A boolean list/array indicating stability at each zero-crossing.
    - zero_crossing_times: The corresponding times of zero-crossing events.

    Returns:
    - mid_times: A list of mid-time positions for each stable region cluster.
    - durations: A list of durations for each stable region cluster.
    """
    mid_times = []
    durations = []
    in_cluster = False  # Track if we're inside a stable cluster
    cluster_start = None

    for i in range(len(stable_regions)):
        if stable_regions[i]:
            if not in_cluster:  # Start a new cluster
                cluster_start = zero_crossing_times[i]
                in_cluster = True
        else:
            if in_cluster:  # End of a cluster
                cluster_end = zero_crossing_times[i - 1]
                mid_times.append((cluster_start + cluster_end) / 2)  # Compute midpoint
                durations.append(cluster_end - cluster_start)  # Compute duration
                in_cluster = False

    # Handle case where the last cluster reaches the end of the array
    if in_cluster:
        cluster_end = zero_crossing_times[-1]
        mid_times.append((cluster_start + cluster_end) / 2)
        durations.append(cluster_end - cluster_start)

    return mid_times, durations

def compute_stable_region_midpoints2(stable_regions, zero_crossing_times):
    """
    Computes the mid-time position of each cluster where stable_region is 1.

    Parameters:
    - stable_regions: A boolean list/array indicating stability at each zero-crossing.
    - zero_crossing_times: The corresponding times of zero-crossing events.

    Returns:
    - mid_times: A list of mid-time positions for each stable region cluster.
    """
    mid_times = []
    in_cluster = False  # Track if we're inside a stable cluster
    cluster_start = None

    for i in range(len(stable_regions)):
        if stable_regions[i]:
            if not in_cluster:  # Start a new cluster
                cluster_start = zero_crossing_times[i]
                in_cluster = True
        else:
            if in_cluster:  # End of a cluster
                cluster_end = zero_crossing_times[i - 1]
                mid_times.append((cluster_start + cluster_end) / 2)  # Compute midpoint
                in_cluster = False

    # Handle case where the last cluster reaches the end of the array
    if in_cluster:
        cluster_end = zero_crossing_times[-1]
        mid_times.append((cluster_start + cluster_end) / 2)

    return mid_times

#plot_spectrogram_plotly(y_use, sr, n_fft=1024, hop_length=512)
#plot_zero_crossing_frequency(y_use, sr, amplitude_threshold=5)
#plot_zero_crossing_frequency_diff(y_use, sr, amplitude_threshold=5)
#plot_zero_crossing_frequency_stable(y_use, sr, amplitude_threshold=5)

## Extract chunk of audio

def Extract_chunk_of_audio(y, sr, time_mid=None, duration_ms=15):
    """
    Extract an audio chunck from the audio data `y` based on the specified times.

    Parameters:
    - y: Audio time series (1D NumPy array).
    - sr: Sampling rate of the audio.
    - time_mid: Mid position of the chunk in seconds (optional).
    - duration_ms: Duration of the chunk to be extracted in milliseconds (optional).

    Returns:
    - truncated_y: The truncated audio data.
    """

    if time_mid is None:
        time_mid = (len(y) / sr) / 2

    # Convert time to sample indices
    start_sample = max(0, int((time_mid - ((duration_ms / 2) / 1000)) * sr))
    end_sample = min(len(y), int((time_mid + ((duration_ms / 2) / 1000)) * sr))

    # Truncate the audio data
    truncated_y = y[start_sample:end_sample]

    return truncated_y

def get_max_amplitude_time(y, sr, duration, n_fft=1024, hop_length=100):
    """
    Finds the time and frequency corresponding to the highest amplitude
    within a restricted time window around the given mid_time.

    Parameters:
    - y: Audio time series (1D numpy array).
    - sr: Sampling rate of the audio.
    - mid_time: The center time around which to search for the max amplitude.
    - duration: The total duration of the search window.
    - n_fft: Number of FFT points (defines frequency resolution).
    - hop_length: Hop length between frames (defines time resolution).

    Returns:
    - max_time_interp: Interpolated time (in seconds) of the highest amplitude pixel within the window.
    - max_freq_interp: Interpolated frequency (in Hz) of the highest amplitude pixel within the window.
    """
    mid_time = (len(y) / sr) / 2

    # Compute the magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='flattop'))

    # Convert to decibels for better contrast
    spec_db = librosa.amplitude_to_db(S, ref=np.max)

    # Get frequency and time values for interpolation
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # Frequency axis
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)  # Time axis

    # Define the time window
    start_time = mid_time - duration / 2
    end_time = mid_time + duration / 2

    # Find indices within the time window
    valid_time_indices = np.where((times >= start_time) & (times <= end_time))[0]

    if len(valid_time_indices) == 0:
        return None  # No valid time indices found in the window

    # Restrict S to the selected time window
    S_window = S[:, valid_time_indices]
    times_window = times[valid_time_indices]

    # Find the index of the maximum amplitude within the window
    max_idx = np.unravel_index(np.argmax(S_window, axis=None), S_window.shape)
    max_freq = freqs[max_idx[0]]
    max_time = times_window[max_idx[1]]

    # Interpolation function
    interp_S = RectBivariateSpline(freqs, times, S)

    # Fine-tune around the detected max index
    fine_freqs = np.linspace(max_freq - sr/n_fft, max_freq + sr/n_fft, 10)
    fine_times = np.linspace(max_time - hop_length/sr, max_time + hop_length/sr, 10)

    # Find the maximum value in the interpolated region
    interp_values = interp_S(fine_freqs, fine_times)
    refined_idx = np.unravel_index(np.argmax(interp_values), interp_values.shape)

    # Get refined time and frequency
    max_freq_interp = float(fine_freqs[refined_idx[0]])
    max_time_interp = float(fine_times[refined_idx[1]])

    max_point = (max_freq_interp, max_time_interp)

    return max_point


def get_max_amplitude_time2(y, sr, duration, n_fft=1024, hop_length=100):
    """
    Finds the time and frequency corresponding to the highest amplitude
    from the FFT process of the signal, using interpolation for better precision.

    Parameters:
    - y: Audio time series (1D numpy array).
    - sr: Sampling rate of the audio.
    - n_fft: Number of FFT points (defines frequency resolution).
    - hop_length: Hop length between frames (defines time resolution).

    Returns:
    - max_time_interp: Interpolated time (in seconds) of the highest amplitude pixel.
    - max_freq_interp: Interpolated frequency (in Hz) of the highest amplitude pixel.
    """

    # Compute the magnitude spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Convert to decibels for better contrast
    spec_db = librosa.amplitude_to_db(S, ref=np.max)

    # Get frequency and time values for interpolation
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # Frequency axis
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)  # Time axis

    # Find the index of the maximum amplitude
    max_idx = np.unravel_index(np.argmax(S, axis=None), S.shape)
    max_freq = freqs[max_idx[0]]
    max_time = times[max_idx[1]]

    # Interpolation function
    interp_S = RectBivariateSpline(freqs, times, S)

    # Fine-tune around the detected max index
    fine_freqs = np.linspace(max_freq - sr/n_fft, max_freq + sr/n_fft, 10)
    fine_times = np.linspace(max_time - hop_length/sr, max_time + hop_length/sr, 10)

    # Find the maximum value in the interpolated region
    interp_values = interp_S(fine_freqs, fine_times)
    refined_idx = np.unravel_index(np.argmax(interp_values), interp_values.shape)

    # Get refined time and frequency
    max_freq_interp = float(fine_freqs[refined_idx[0]])
    max_time_interp = float(fine_times[refined_idx[1]])

    max_point = (max_freq_interp, max_time_interp)

    return max_point

def interpolate_spectrogram(S):
    """
    Interpolates the spectrogram S for smooth amplitude estimation.

    Parameters:
    - S: 2D NumPy array representing the magnitude spectrogram.

    Returns:
    - interpolated_function: A function that takes fractional (freq_idx, time_idx)
      and returns interpolated amplitude values.
    """
    def interpolated_function(freq_idx, time_idx):
        coords = np.array([[freq_idx], [time_idx]])  # Shape (2, 1)
        return scipy.ndimage.map_coordinates(S, coords, order=1, mode='reflect')[0]

    return interpolated_function


def sample_line_from_max_amp_dynamic(S, freqs, times, sample_point, sr, slope=1, amplitude_threshold_percent=15, nb_step=50):
    """
    Samples amplitude values along a line through the maximum amplitude point in a spectrogram,
    using frequency and time coordinates instead of raw indices.

    The line extends in both directions until the interpolated amplitude is <= amplitude_threshold_fraction * (max amplitude)
    or until a maximum number of points is reached in that direction.

    The line is defined by:
        freq = max_freq + t * slope
        time = max_time + t
    where t is a parameter (with a step size of `step`).

    Parameters:
    - S: 2D numpy array (spectrogram) with shape (frequency_bins, time_frames).
    - freqs: 1D numpy array of frequency values corresponding to rows of S.
    - times: 1D numpy array of time values corresponding to columns of S.
    - max_freq: The frequency coordinate of the max amplitude point.
    - max_time: The time coordinate of the max amplitude point.
    - slope: The slope (vertical displacement per horizontal unit) of the line. Default is 2.
    - amplitude_threshold_fraction: Fraction of the maximum amplitude at which to stop sampling.
                                    Default is 0.1 (i.e. 10% of the maximum).
    - step: Step size along the line (in units of time increments). Default is 0.5.

    Returns:
    - t_line: 1D numpy array of t values for each sampled point (ordered from negative to positive).
    - amplitude_line: 1D numpy array of interpolated amplitude values along the line.
    """
    if isinstance(slope, (int, float)) and not np.isnan(slope):
        if np.isinf(slope):
            return None, None
    else:
        return None, None


    freq_to_sample, time_to_sample = sample_point

    # Interpolation function for spectrogram
    interp_S = scipy.interpolate.RectBivariateSpline(freqs, times, S)

    # Get the maximum amplitude value at (max_freq, max_time)
    amp_max = interp_S(freq_to_sample, time_to_sample)[0][0]
    threshold_value = amplitude_threshold_percent * amp_max /100

    # Define the maximum number of points to sample in each direction.
    max_count = 500

    step = (1/sr*100)/nb_step

    real_slope = slope * 3730000
    #f 2 y pixels/x pixel (the equivalent of 3.73 kHz/ms) 3.73 10^6 Hz/s

    # Sampling in the positive direction
    pos_ts, pos_amplitudes = [], []
    t, pos_count = 0.0, 0

    while True:
        freq = freq_to_sample + t * real_slope
        time = time_to_sample + t
        amplitude = interp_S(freq, time)[0][0]

        if t != 0 and amplitude <= threshold_value:
            #print(amplitude)
            break

        pos_ts.append(t)
        pos_amplitudes.append(amplitude)

        pos_count += 1
        if pos_count >= max_count:
            #print(pos_count)
            break

        t += step / np.sqrt(1 + slope ** 2)

    # Sampling in the negative direction
    neg_ts, neg_amplitudes = [], []
    t, neg_count = -step, 0
    while True:
        freq = freq_to_sample + t * real_slope
        time = time_to_sample + t
        amplitude = interp_S(freq, time)[0][0]

        if amplitude <= threshold_value:
            #print(amplitude)
            break

        neg_ts.append(t)
        neg_amplitudes.append(amplitude)

        neg_count += 1
        if neg_count >= max_count:
            #print(neg_count)
            break

        t -= step / np.sqrt(1 + slope ** 2)

    # Reverse negative direction lists for proper order
    neg_ts.reverse()
    neg_amplitudes.reverse()

    # Combine negative and positive parts
    t_line = np.array(neg_ts + pos_ts)
    amplitude_line = np.array(neg_amplitudes + pos_amplitudes)

    return t_line, amplitude_line

import numpy as np

def gaussian2(t, A, mu, sigma):
    """Optimized Gaussian function."""
    sigma = np.maximum(sigma, 1e-10)  # Ensure sigma is not too small
    sigma_sq = 2 * np.square(sigma)  # Precompute 2 * sigma^2
    exponent = np.clip(-np.square(t - mu) / sigma_sq, -100, 100)  # Compute exponent efficiently
    return A * np.exp(exponent)

import numpy as np
import scipy.optimize

def fit_gaussian2(t_line, amplitude_line):
    """Optimized version of fit_gaussian to improve execution speed."""

    # Find max value & index
    max_index = np.argmax(amplitude_line)
    max_value = amplitude_line[max_index]
    half_max = max_value / 2  # Precompute to avoid redundant division

    # Use np.searchsorted() to find left/right limits efficiently
    left_limit = np.searchsorted(amplitude_line[:max_index][::-1], half_max, side='right')
    left_limit = max_index - left_limit  # Convert from reversed index

    right_limit = np.searchsorted(amplitude_line[max_index:], half_max, side='right')
    right_limit = max_index + right_limit  # Convert to original index

    # Extract the subset of points
    t_selected = t_line[left_limit:right_limit + 1]
    amp_selected = amplitude_line[left_limit:right_limit + 1]

    # Ensure at least 3 points
    if len(t_selected) < 3:
        return None, None  # Avoid allocating np.zeros_like

    # Initial parameter guess
    A_guess = max_value
    mu_guess = t_line[max_index]
    sigma_guess = (t_selected[-1] - t_selected[0]) / 4.0
    p0 = [A_guess, mu_guess, sigma_guess]

    # try:
    #     # Fit Gaussian with a reduced maxfev
    #     popt, _ = scipy.optimize.curve_fit(gaussian, t_selected, amp_selected, p0=p0, maxfev=500)
    #     return popt, gaussian(t_line, *popt)
    # except (RuntimeError, ValueError):
    #     return None, None  # Avoid unnecessary allocations
    # Adaptive maxfev
    maxfev_values = [100, 500, 1000, 5000]  # Try increasing values
    for maxfev in maxfev_values:
        try:
            popt, pcov = scipy.optimize.curve_fit(gaussian, t_selected, amp_selected, p0=p0, maxfev=maxfev)
            print(maxfev)
            return popt, gaussian(t_line, *popt)
        except RuntimeError:
            print(f"⚠️ Fit failed with maxfev={maxfev}, increasing maxfev...")

    print("❌ Curve fitting failed after all attempts.")
    return None, np.zeros_like(t_line)

def gaussian(t, A, mu, sigma):
    """Gaussian function."""
    sigma = max(sigma, 1e-10)  # Set a minimum value to avoid numerical issues
    exponent = -((t - mu) ** 2) / (2 * sigma ** 2)
    exponent = np.clip(exponent, -100, 100)  # Prevent underflow
    return A * np.exp(exponent)


def fit_gaussian(t_line, amplitude_line):
    """
    Fits a Gaussian to the provided amplitude data, considering only values
    where amplitude_line is between max and max/2.

    Parameters:
    - t_line: 1D numpy array of t values.
    - amplitude_line: 1D numpy array of amplitude values.

    Returns:
    - popt: Optimal values for the parameters (A, mu, sigma), or None if fitting fails.
    - gaussian_fit: The fitted Gaussian values evaluated at t_line, or zeros if fitting fails.
    """
    # Find the maximum value and its index
    max_value = np.max(amplitude_line)
    max_index = np.argmax(amplitude_line)
    half_max = max_value / 2

    # Find the left and right limits where amplitude_line falls below max/2
    left_limit, right_limit = max_index, max_index

    while left_limit > 0 and amplitude_line[left_limit] >= max_value / 2:
        left_limit -= 1
    while right_limit < len(amplitude_line) - 1 and amplitude_line[right_limit] >= max_value / 2:
        right_limit += 1

    # # Use np.searchsorted() to find left/right limits efficiently
    # left_limit = np.searchsorted(amplitude_line[:max_index][::-1], half_max, side='right')
    # left_limit = max_index - left_limit  # Convert from reversed index

    # right_limit = np.searchsorted(amplitude_line[max_index:], half_max, side='right')
    # right_limit = max_index + right_limit  # Convert to original index

    # Extract the subset of points within the identified limits
    t_selected = t_line[left_limit:right_limit + 1]
    amp_selected = amplitude_line[left_limit:right_limit + 1]

    # Ensure we have at least 3 points for fitting
    if len(t_selected) < 3:
        #print("Not enough points for Gaussian fit")
        return None, np.zeros_like(t_line)

    # Initial parameter guess
    A_guess = max_value
    mu_guess = t_line[max_index]
    sigma_guess = (t_selected[-1] - t_selected[0]) / 4.0
    p0 = [A_guess, mu_guess, sigma_guess]

    try:
        # Fit the Gaussian function to the selected data
        popt, _ = scipy.optimize.curve_fit(gaussian, t_selected, amp_selected, p0=p0, maxfev=500)
        # Evaluate the fitted Gaussian on the full t_line range
        gaussian_fit = gaussian(t_line, *popt)
        return popt, gaussian_fit
    except (RuntimeError, ValueError):
        # If fitting fails, return None for parameters and a zero array for the Gaussian fit
        return None, np.zeros_like(t_line)

    # # Adaptive maxfev
    # maxfev_values = [10, 20, 30, 40, 50, 100, 500]  # Try increasing values
    # for maxfev in maxfev_values:
    #     try:
    #         popt, pcov = scipy.optimize.curve_fit(gaussian, t_selected, amp_selected, p0=p0, maxfev=maxfev)
    #         #print(f"✅ Curve fit successful with maxfev={maxfev}")
    #         print(maxfev)
    #         return popt, gaussian(t_line, *popt)
    #     except RuntimeError:
    #         #print(f"⚠️ Fit failed with maxfev={maxfev}, increasing maxfev...")
    #         print(' ')

    # print("❌ Curve fitting failed after all attempts.")
    # return None, np.zeros_like(t_line)

def calculate_percentage_variation(t_line, amplitude_line, fit_function=fit_gaussian, slope=1):
    """
    Calculates the percentage variation between the fitted Gaussian and the amplitude_line,
    and returns the distance along t_line between the maximum of the fitted Gaussian and
    the maximum of the original amplitude data.

    If the fitting fails, it returns a `percent_variation` of 100.

    Parameters:
    - t_line: 1D numpy array of t values.
    - amplitude_line: 1D numpy array of amplitude values.
    - fit_function: Function to use for Gaussian fitting. It should accept (t_line, amplitude_line)
      and return (popt, gaussian_fit). Default is fit_gaussian.

    Returns:
    - percent_variation: The percentage variation of the fitted Gaussian compared to amplitude_line.
    - max_gaussian_value: The maximum value of the Gaussian fit.
    - max_distance_gauss: The position of the maximum of the Gaussian fit.
    - popt: Optimized parameters of the Gaussian function. Returns `None` if fitting fails.
    """
    if t_line is None or amplitude_line is None:
        #print("t_line is None or amplitude_line is None")
        return 100, 0, None, None, None  # Return early if input is invalid

    try:
        popt, gaussian_fit = fit_function(t_line, amplitude_line)
    except (RuntimeError, ValueError, scipy.optimize.OptimizeWarning):
        return 100, 0, None, None, None  # Fitting failed, return high variation

    # Compute the Gaussian max value
    # if gaussian_fit is None:
    #     return 100, 0, None, None, None

    max_gauss = np.max(gaussian_fit)

    # Create a mask for values where the Gaussian is between max and max/2
    mask = (gaussian_fit >= (np.max(gaussian_fit) / 2))

    # Compute the absolute error only at selected points
    absolute_error = np.abs(amplitude_line[mask] - gaussian_fit[mask])

    # Compute the percentage variation considering only the selected points
    percent_variation = (np.sum(absolute_error) / np.sum(np.abs(amplitude_line[mask]))) * 100

    # Find the corresponding t values where amplitude and Gaussian fit reach their maximum
    t_max_gaussian = t_line[np.where(gaussian_fit == max_gauss)][0]

    # Compute the absolute distance along t_line.
    max_distance_gauss = t_max_gaussian  # Now directly the time value
    max_freq_gauss = t_max_gaussian * slope * 3730000

    # Maximum value of the Gaussian fit (z value)
    max_gaussian_value = max_gauss

    return percent_variation, max_gaussian_value, max_distance_gauss,  popt, max_freq_gauss

import numpy as np
import scipy.optimize

def calculate_percentage_variation3(t_line, amplitude_line, fit_function=fit_gaussian, slope=1):
    """Optimized function to reduce execution time."""

    if t_line is None or amplitude_line is None:
        return 100, 0, None, None, None  # Return early if input is invalid

    try:
        popt, gaussian_fit = fit_function(t_line, amplitude_line)
        if gaussian_fit is None:
            return 100, 0, None, None, None  # Fitting failed
    except (RuntimeError, ValueError, scipy.optimize.OptimizeWarning):
        return 100, 0, None, None, None  # Fitting failed, return high variation

    # Store max value of Gaussian fit to avoid redundant computation
    max_gauss = np.max(gaussian_fit)

    # Find the maximum index of the Gaussian fit using argmax (faster than np.where)
    max_index_gauss = np.argmax(gaussian_fit)
    t_max_gaussian = t_line[max_index_gauss]

    # Use np.nonzero() instead of applying the mask twice
    mask_indices = np.nonzero(gaussian_fit >= max_gauss / 2)[0]

    if len(mask_indices) == 0:  # Edge case handling
        return 100, max_gauss, t_max_gaussian, popt, t_max_gaussian * slope * 3730000

    # Compute absolute error only at selected points
    absolute_error = np.abs(amplitude_line[mask_indices] - gaussian_fit[mask_indices])

    # Compute percentage variation using optimized sum calculation
    percent_variation = (absolute_error.sum() / np.abs(amplitude_line[mask_indices]).sum()) * 100

    # Compute frequency using slope
    max_freq_gauss = t_max_gaussian * slope * 3730000

    return percent_variation, max_gauss, t_max_gaussian, popt, max_freq_gauss

def point_along_line(initial_point, slope, distance):
    """
    Returns the coordinate (as a tuple) of a point along a line passing through an initial point,
    given the slope of the line and the horizontal displacement (distance along the x-axis)
    from the initial point.

    Parameters:
    - initial_point: Tuple (x0, y0) representing the starting coordinate.
    - slope: Slope (m) of the line (rise over run).
    - distance: Horizontal distance (d) from the initial point along the x-axis.
                A positive value moves in the direction of increasing x,
                and a negative value moves in the opposite direction.

    Returns:
    - point: Tuple (x, y) representing the coordinate of the new point.
    """
    if (initial_point is None or slope is None or distance is None):
        #print("initial_point is None or slope is None or distance is None")
        return None

    # Unpack the initial point.
    y0, x0 = initial_point

    #real_slope = slope * sr * sr / (n_fft * hop)
    real_slope = slope * 3730000

    # Since 'distance' is the horizontal displacement:
    dx = distance
    dy = real_slope * distance

    # Compute the new point.
    return (y0 + dy, x0 + dx)

def initial_call_trend(yt, sr, duration, step_ns=50):
    """
    Computes the spectrogram of the given audio signal y, finds the maximum amplitude index,
    and analyzes the trend using Gaussian fit error metrics along sampled amplitude lines.

    Returns:
        A NumPy array where each row corresponds to a candidate index and contains:
        - max_time (float): Adjusted time coordinate of the maximum Gaussian fit.
        - max_freq (float): Adjusted frequency coordinate of the maximum Gaussian fit.
        - max_gauss_z (float): Maximum value of the Gaussian fit.
        - percent_variation (float): Percentage variation between the data and the Gaussian fit.
        - max_distance (float): Distance along t_line between the maximum positions.
        - slope (float): Fixed slope value used for the line sampling.
    """
    # Compute spectrogram
    Sp = np.abs(librosa.stft(yt, n_fft=1024, hop_length=100, window='flattop'))

    # Frequency and time axes
    freqss = librosa.fft_frequencies(sr=sr, n_fft=1024)
    timess = librosa.frames_to_time(np.arange(Sp.shape[1]), sr=sr, hop_length=100)

    # Find the max amplitude point
    max_freq, max_time = get_max_amplitude_time(yt, sr, duration=duration)
    #print(max_freq, max_time)

    step = step_ns/1000000

    # Define candidate indices (time axis adjustments)
    time_offsets = [-3*step,-2*step, -1*step, 0*step, 1*step, 2*step, 3*step]
    candidate_timess = max_time + np.array(time_offsets)
    candidate_freqss = np.full_like(candidate_timess, max_freq)

    # Initialize results array
    results = np.zeros((len(candidate_timess), 9))  # 9 columns for different computed values

    for i, (candidate_time, candidate_freq) in enumerate(zip(candidate_timess, candidate_freqss)):
        # Sample amplitude values along a line through the candidate point
        candidate_point = (candidate_freq, candidate_time)  # Create a tuple (freq, time)

        t_line, amplitude_line = sample_line_from_max_amp_dynamic(Sp, freqss, timess, candidate_point, sr)

        #plot_amplitude_line_with_gaussian(t_line, amplitude_line)

        # Compute Gaussian fit metrics
        percent_variation, max_gauss_z, max_dist_gauss, popt, Freq_from_max_gauss = calculate_percentage_variation(t_line, amplitude_line, slope = 1)

        # Compute the adjusted coordinates using the max distance along the line
        max_gauss_freq, max_gauss_time  = point_along_line(candidate_point, 1, max_dist_gauss)

        # Store results in the array
        if popt is not None and len(popt) >= 3 and None not in (max_gauss_time, max_gauss_freq, max_gauss_z, percent_variation, max_dist_gauss):
          results[i] = [max_gauss_time, max_gauss_freq, max_gauss_z, percent_variation, max_dist_gauss, popt[0], popt[1], popt[2], 1]

    #print(results)
    #plot_initial_trend(Sp, candidate_freqss, candidate_timess, results)

    return results

def validate_results(results):
    """
    Validate the results dictionary.

    Each candidate's results are valid if:
      - percent_variation < 30, and
      - max_distance < 2.

    Parameters:
    - results: Dictionary where each key is a candidate index (tuple) and each value is a dictionary with keys:
          "max_gauss", "percent_variation", and "max_distance".

    Returns:
    - all_valid: Boolean indicating if all candidate results are valid.
    - validation_dict: Dictionary mapping each candidate index to a boolean (True if valid, False otherwise).
    """
    validation_dict = {}
    all_valid = True

    for candidate, metrics in results.items():
        valid = (metrics["percent_variation"] < 30) and (metrics["max_distance"] < 2)
        validation_dict[candidate] = valid
        if not valid:
            all_valid = False

    return all_valid, validation_dict

def fit_polynomial(candidate_points, smoothing_factor=1, degree=2):
    """
    Fits a polynomial regression to candidate points.

    Parameters:
    - candidate_points: Numpy array of (x, y) points.
    - smoothing_factor: Not used here, kept for compatibility.
    - degree: Degree of the polynomial.

    Returns:
    - poly_func: A callable polynomial function.
    - x_smooth: A range of x values for plotting.
    - y_smooth: Corresponding y values of the polynomial fit.
    """
    candidate_points = np.asarray(candidate_points, dtype=float)
    if candidate_points.ndim != 2 or candidate_points.shape[1] < 2:
        return None, None, None

    x, y = candidate_points[:, 0], candidate_points[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    # Repeated time coordinates make np.polyfit divide by a zero scale and
    # produce the RuntimeWarning seen during directory processing.  Keep one
    # frequency value (their mean) for each distinct time coordinate.
    if x.size:
        unique_x, inverse = np.unique(x, return_inverse=True)
        if unique_x.size != x.size:
            sums = np.bincount(inverse, weights=y)
            counts = np.bincount(inverse)
            x, y = unique_x, sums / counts

    if x.size <= degree or np.ptp(x) <= np.finfo(float).eps:
        return None, None, None

    try:
        # Centering/scaling x avoids a poorly conditioned fit for times in the
        # microsecond range. Convert the result back to the usual power basis.
        fitted = np.polynomial.Polynomial.fit(x, y, degree).convert()
        poly_coeffs = fitted.coef[::-1]
        poly_func = np.poly1d(poly_coeffs)

        # Generate smooth curve for plotting
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = poly_func(x_smooth)

        return poly_func, x_smooth, y_smooth
    except Exception as e:
        #print("Polynomial fitting failed:", e)
        return None, None, None

def get_extrapolated_points(trend_results, step_ns=40, LeftRight=0, nb_point_to_consider=8):
    """
    Returns an extrapolated point on the smoothing spline curve along with the slope
    of the perpendicular line at this point.

    Parameters:
    - trend_results: List of lists containing extracted data points.
      Each entry: [y_x, max_gauss_z, percent_variation, max_distance, popt, slope].
    - smoothing_factor: Smoothing factor for UnivariateSpline.
    - step_ns: Step size along the slope direction.
    - LeftRight: 0 to extrapolate before the first point, 1 to extrapolate after the last point.

    Returns:
    - new_point: (y_new, x_new) Extrapolated point in float precision.
    - slope_perp: Slope of the perpendicular line at the extrapolated point.
    """

    step = step_ns / 1e6  # Convert ns to seconds

    # Extract and sort (time, frequency) points from trend_results
    candidate_points = np.array([(entry[0], entry[1]) for entry in trend_results])

    if candidate_points is None:
        print("Error: Not enough points for spline fitting")
        return None, None

    if len(candidate_points) < 4 or candidate_points is None:
        print("Error: Not enough points for spline fitting")
        return None, None

    # If more than nb_point_to_consider points, keep only the first or last nb_point_to_consider based on LeftRight
    if len(candidate_points) > nb_point_to_consider:
        candidate_points = candidate_points[:nb_point_to_consider] if LeftRight == 0 else candidate_points[-nb_point_to_consider:]

    # Fit a smoothing spline
    try:
        spline, x_smooth, y_smooth = fit_polynomial(candidate_points, degree=1)
    except Exception as e:
        print("Spline fitting failed:", e)
        return None, None

    if spline is None:
        return None, None

    # Determine the extrapolation reference point
    if LeftRight == 0:
        ref_x = candidate_points[0, 0]  # First point
        ref_y = candidate_points[0, 1]
    else:
        ref_x = candidate_points[-1, 0] # Last point
        ref_y = candidate_points[-1, 1]

    # Compute slope using polynomial derivative
    try:
        poly_derivative = np.polyder(spline)  # Compute the derivative of the polynomial
        slope = poly_derivative(ref_x)  # Evaluate it at ref_x
        slope = slope / 3730000
    except Exception as e:
        #print("Derivative computation failed:", e)
        return ref_y, None

    # Compute the step in x and y along the slope direction
    if abs(slope) < 1e-10:
        dx, dy = 0, step  # Move purely in y direction if the slope is too small
    else:
        norm_factor = np.sqrt(1 + slope**2)  # Normalize movement along the slope
        dx = (step / norm_factor) * (-1 if LeftRight == 0 else 1)  # Adjust sign based on extrapolation direction
        dy = (slope * dx * 3730000)  # Keep consistent with actual slope direction

        # Determine extrapolated point
    new_x = ref_x + dx
    new_y = ref_y + dy

    # Compute slope of the perpendicular line
    slope_perp = float('inf') if abs(slope) < 1e-10 else -1 / slope

    new_point = (new_y, new_x)

    #if new_y > 70000:
      #plot_trend_analysis(candidate_points, new_point, slope_perp)
      #print(f"Reference point: ({ref_x}, {ref_y})")
      #print(f"Extrapolated point: ({new_x}, {new_y})")
      #print(f"dx,dy: ({dx}, {dy})")

    return new_point, slope_perp

def calculate_angle_between_lines(trend_results, nb_points=14, LeftRight=0):
    """
    Computes the angle between two lines, each fitted to a subset of points.

    Parameters:
    - trend_results: List of lists containing extracted data points.
      Each entry: [x, y, z, percent_variation, max_dist_gauss, popt[0], popt[1], popt[2], 2].
    - nb_points: Minimum number of points required to calculate the angle.
    - LeftRight: Determines which section to analyze.
      0 -> First 4 points and the next (nb_points - 4) points.
      1 -> Last 4 points and the previous (nb_points - 4) points.

    Returns:
    - angle: Absolute angle in degrees between the two fitted lines, or None if not enough points.
    """

    # Ensure we have enough points
    if len(trend_results) < nb_points:
        return 0  # Not enough points to compute angle

    # Extract and sort candidate points
    candidate_points = np.array(sorted([(entry[0], entry[1]) for entry in trend_results]))

    x_vals = candidate_points[:, 0]
    y_vals = candidate_points[:, 1]

    # Select points for each line based on LeftRight
    half_nb_points = nb_points // 2  # Divide points roughly in half
    if LeftRight == 0:
        if len(x_vals) < half_nb_points + 4:  # Ensure valid slicing
            return 0  # Not enough points for valid regression
        x1, y1 = x_vals[:4], y_vals[:4]  # First 4 points
        x2, y2 = x_vals[4:nb_points], y_vals[4:nb_points]  # Next (nb_points - 4) points
    else:
        if len(x_vals) < nb_points:  # Ensure valid slicing
            return 0  # Not enough points for valid regression
        x1, y1 = x_vals[-4:], y_vals[-4:]  # Last 4 points
        x2, y2 = x_vals[-nb_points:-4], y_vals[-nb_points:-4]  # Preceding (nb_points - 4) points

    # Ensure x1 and x2 are not empty
    if len(x1) < 2 or len(x2) < 2:
        return 0  # Not enough points for fitting

    # Use the guarded fitter: initial segments can also contain duplicate times.
    line1, _, _ = fit_polynomial(np.column_stack((x1, y1)), degree=1)
    line2, _, _ = fit_polynomial(np.column_stack((x2, y2)), degree=1)
    if line1 is None or line2 is None:
        return 0

    slope1 = line1.c[0] / 3730000
    slope2 = line2.c[0] / 3730000

    # Compute the angle between the two slopes
    angle_rad = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def add_new_point_to_results(results, new_point, LeftRight=0):
    """
    Adds a new point to the filtered results from extract_relevant_data.

    Parameters:
    - results: Dictionary containing candidate points and their corresponding data.
    - new_point: List containing the values [y_x, max_gauss_z, percent_variation, max_distance, popt, slope].
    - add_at_start: Boolean flag to specify if the new point should be added at the beginning (True) or the end (False).

    Returns:
    - updated_results: List of lists, updated with the new point.
    """
    filtered_results = results.tolist()  # Extract the data into a list

    if (LeftRight == 0):
        # Insert the new point at the beginning
        filtered_results.insert(0, new_point)

    else:
        # Append the new point at the end
        filtered_results.append(new_point)


    return np.array(filtered_results)


def extract_relevant_data(results):
    """
    Extracts specific data from results and stores them in an array without keys.

    Parameters:
    - results: Dictionary containing candidate points and their corresponding data.

    Returns:
    - filtered_results: List of lists, where each inner list contains only the relevant values.
    """
    filtered_results = []  # Initialize an empty list

    for data in results.values():
        # Extract only the necessary values
        y_x = data["max_gauss"]  # (y, x)
        max_gauss_z = data["max_gauss_z"]
        percent_variation = data["percent_variation"]
        max_distance = data["max_distance"]
        popt = data["popt"]
        slope = data["slope"]

        # Append values as a list (removing dictionary structure)
        filtered_results.append([y_x, max_gauss_z, percent_variation, max_distance, popt, slope])

    return filtered_results  # Returns a list of lists

def remove_points_from_results(results, n=4, LeftRight=0):
    """
    Removes `n` points from the filtered results.

    Parameters:
    - results: NumPy array containing candidate points and their corresponding data.
    - n: Number of points to remove.
    - LeftRight: Integer flag (0 for removing from the beginning, 1 for removing from the end).

    Returns:
    - updated_results: NumPy array with `n` points removed.
    """
    #print('results')
    #print(results)
    filtered_results = results.tolist()  # Convert to a list for modification

    if not filtered_results:  # Ensure the list is not empty
        return np.array([])

    if LeftRight == 0:
        # Remove `n` points from the beginning
        filtered_results = filtered_results[n:]
    else:
        # Remove `n` points from the end
        filtered_results = filtered_results[:-n] if n < len(filtered_results) else []
    #print('results filtered')
    #print(np.array(filtered_results))
    return np.array(filtered_results)

def smooth_trend_spline(trend_results, smoothing_factor=1):
    """
    Approximates the trend points from initial_call_trend using Smoothing Splines.

    Parameters:
    - trend_results: Dictionary with candidate indices as keys and max_gauss coordinates as values.
    - smoothing_factor: Controls the smoothness of the spline (higher values give smoother curves).

    Returns:
    - spline_func: A function that interpolates frequency values given time.
    - times_smooth: Array of smoothed time values.
    - freqs_smooth: Array of smoothed frequency values.
    """
    # Extract max_gauss points (time and frequency coordinates)
    times = []
    freqs = []

    for candidate, metrics in trend_results.items():
        max_gauss = metrics["max_gauss"]  # (frequency_index, time_index)
        times.append(max_gauss[1])
        freqs.append(max_gauss[0])

    times = np.array(times)
    freqs = np.array(freqs)

    # Sort by time to avoid issues in interpolation
    sorted_indices = np.argsort(times)
    times = times[sorted_indices]
    freqs = freqs[sorted_indices]

    # Fit a smoothing spline
    spline_func = interp.UnivariateSpline(times, freqs, s=smoothing_factor)

    # Generate a smooth curve
    times_smooth = np.linspace(times.min(), times.max(), 300)
    freqs_smooth = spline_func(times_smooth)

    return spline_func, times_smooth, freqs_smooth


def smooth_trend_spline2(trend_results, smoothing_factor=1):
    """
    Approximates the trend points from trend_results using Smoothing Splines.

    Parameters:
    - trend_results: List of lists containing extracted data points.
      Each entry: [y_x, max_gauss_z, percent_variation, max_distance, popt, slope].
    - smoothing_factor: Controls the smoothness of the spline (higher values give smoother curves).

    Returns:
    - spline_func: A function that interpolates frequency values given time.
    - times_smooth: Array of smoothed time values.
    - freqs_smooth: Array of smoothed frequency values.
    """
    # Extract max_gauss points (time and frequency coordinates)
    times = []
    freqs = []

    for entry in trend_results:
        y_x = entry[0]  # max_gauss (y, x)
        times.append(y_x[1])  # x-coordinate (time)
        freqs.append(y_x[0])  # y-coordinate (frequency)

    times = np.array(times)
    freqs = np.array(freqs)

    # Sort by time to avoid issues in interpolation
    sorted_indices = np.argsort(times)
    times = times[sorted_indices]
    freqs = freqs[sorted_indices]

    # Fit a smoothing spline
    spline_func = UnivariateSpline(times, freqs, s=smoothing_factor)

    # Generate a smooth curve
    times_smooth = np.linspace(times.min(), times.max(), 300)
    freqs_smooth = spline_func(times_smooth)

    return spline_func, times_smooth, freqs_smooth

def process_side(Seven_points, Spectro, freqs, times, sr, LeftRight, max_amplitude, Previous_Curve=None,
                 max_iterations=500, min_time_progress=1e-9):
    """
    Processes one side (Left or Right) for Gaussian variation analysis.

    Parameters:
    - Seven_points: Data structure containing extracted information.
    - Spectro: Spectrogram data.
    - freqs: Frequency values.
    - times: Time values.
    - LeftRight: Direction of processing (0 for left, 1 for right).
    - max_amplitude: Maximum amplitude value.
    - Previous_Curve: Previous curve data (optional).

    Returns:
    - Updated Seven_points after adding new points.
    """

    if Seven_points is None:
        return None

    # Determine min and max x-coordinates from Previous_Curve
    if Previous_Curve is not None:
        min_previous_x = min(point[0] for point in Previous_Curve)
        max_previous_x = max(point[0] for point in Previous_Curve)
    else:
        min_previous_x, max_previous_x = float('inf'), float('-inf')

    # Compute initial extrapolated points and sample amplitude line
    NewPointExtrapolated, NewPointExtrapolatedSlope = get_extrapolated_points(
        Seven_points, LeftRight=LeftRight, nb_point_to_consider=20
    )

    if NewPointExtrapolated is None:
        return None

    t_line3, amplitude_line3 = sample_line_from_max_amp_dynamic(
        Spectro, freqs, times, NewPointExtrapolated, sr, slope=NewPointExtrapolatedSlope
    )

    if t_line3 is None:
        return None


    #pdb.set_trace()

    if t_line3.size < 10:
        return Seven_points

    # Compute Gaussian variation metrics
    Gaussian_variation_percent, Gauss_z_value, Distance_from_max_gauss, Gaussian_function, Freq_from_max_gauss = (
        calculate_percentage_variation(t_line3, amplitude_line3, slope=NewPointExtrapolatedSlope)
    )


    #pdb.set_trace()

    i, AngleOfTrend = 0, 0
    previous_Max_Gauss_coordinate = None  # Track previous Gaussian max coordinate

    Max_Gauss_coordinate = point_along_line(NewPointExtrapolated, NewPointExtrapolatedSlope, Distance_from_max_gauss)
    Gauss_y_value, Gauss_x_value = Max_Gauss_coordinate
    limit_Gauss_x_value = Gauss_x_value + 1 if LeftRight == 0 else Gauss_x_value - 1

    #print('process side 1')
    #pdb.set_trace()
    # Iteratively refine Gaussian max point

    #pdb.set_trace()
    while (
        Gaussian_variation_percent < 30 and
        abs(Distance_from_max_gauss) < 0.00069 and
        ((Gauss_z_value > max_amplitude/20 and LeftRight==0) or (Gauss_z_value > max_amplitude/20 and LeftRight==1)) and
        i < max_iterations and
        abs(Freq_from_max_gauss) < 1300 and
        AngleOfTrend < 50 and
        (previous_Max_Gauss_coordinate is None or Max_Gauss_coordinate != previous_Max_Gauss_coordinate or i < 2) and
        ((LeftRight == 0 and Gauss_x_value < min_previous_x) or (LeftRight == 1 and Gauss_x_value > max_previous_x)) and
        ((LeftRight == 0 and Gauss_x_value <= limit_Gauss_x_value + 0.00002) or
         (LeftRight == 1 and Gauss_x_value >= limit_Gauss_x_value - 0.00002)) and
        Gauss_x_value > 0 and Gauss_x_value < 0.015
    ):
        i += 1

        previous_Max_Gauss_coordinate = Max_Gauss_coordinate  # Store previous coordinate

        #Update the extrem x value
        if (Gauss_x_value < limit_Gauss_x_value and LeftRight == 0) or (Gauss_x_value > limit_Gauss_x_value and LeftRight == 1):
            limit_Gauss_x_value = Gauss_x_value

        #Get coordinate of Gauss max
        Max_Gauss_coordinate = point_along_line(NewPointExtrapolated, NewPointExtrapolatedSlope, Distance_from_max_gauss)

        if Max_Gauss_coordinate is None:
            break

        Gauss_y_value, Gauss_x_value = Max_Gauss_coordinate

        # A Gaussian correction can bring the new point back onto the current
        # endpoint. Inserting it again creates duplicate x values, so the next
        # extrapolation never makes real progress. Require monotonic movement.
        edge_x = float(Seven_points[0][0] if LeftRight == 0 else Seven_points[-1][0])
        made_progress = (
            Gauss_x_value < edge_x - min_time_progress
            if LeftRight == 0
            else Gauss_x_value > edge_x + min_time_progress
        )
        if not made_progress:
            break

        #Add the new point to the trend
        new_point = [Gauss_x_value, Gauss_y_value, Gauss_z_value, Gaussian_variation_percent,
                     Distance_from_max_gauss, Gaussian_function[0], Gaussian_function[1],
                     Gaussian_function[2], NewPointExtrapolatedSlope]
        Seven_points = add_new_point_to_results(Seven_points, new_point, LeftRight=LeftRight)

        #Get a new extrapolated point
        NewPointExtrapolated, NewPointExtrapolatedSlope = get_extrapolated_points(
            Seven_points, LeftRight=LeftRight, nb_point_to_consider=20
        )

        if NewPointExtrapolated is None or NewPointExtrapolatedSlope is None:
            break

        #Sample amplitude at this new point
        t_line3, amplitude_line3 = sample_line_from_max_amp_dynamic(
        Spectro, freqs, times, NewPointExtrapolated, sr, slope=NewPointExtrapolatedSlope
        )

        if t_line3 is None or amplitude_line3 is None or t_line3.size < 10:
            break

        #Get data on amplitude sample at this new point
        Gaussian_variation_percent, Gauss_z_value, Distance_from_max_gauss, Gaussian_function, Freq_from_max_gauss = (
            calculate_percentage_variation(t_line3, amplitude_line3, slope=NewPointExtrapolatedSlope)
        )
        AngleOfTrend = calculate_angle_between_lines(Seven_points, LeftRight)

    #print('break 2')
    #pdb.set_trace()
    #print('process side 2')
    #pdb.set_trace()

    # print(f'LeftRight: {LeftRight}, Iterations: {i}')

    # # Error handling and corrective actions
    # if AngleOfTrend >= 50:
    #     Seven_points = remove_points_from_results(Seven_points, n=5, LeftRight=LeftRight)
    #     print(f'Error: AngleOfTrend {AngleOfTrend} exceeded limit. Deleted 5 points.')

    # if (LeftRight == 0 and Gauss_x_value > limit_Gauss_x_value + 0.00002) or (LeftRight == 1 and Gauss_x_value < limit_Gauss_x_value - 0.00002):
    #     Seven_points = remove_points_from_results(Seven_points, n=2, LeftRight=LeftRight)
    #     print(f'Error: Gauss_x_value {Gauss_x_value} exceeded limit {limit_Gauss_x_value}. Deleted 2 points.')

    # if Max_Gauss_coordinate == previous_Max_Gauss_coordinate and i > 2:
    #     print(f'Warning: Max_Gauss_coordinate did not change: {Max_Gauss_coordinate}')

    # if Gaussian_variation_percent >= 30:
    #     print('Error: Gaussian_variation_percent exceeded threshold. Plotting results. ')
    #     #plot_amplitude_line_with_gaussian(t_line3, amplitude_line3)

    # if abs(Distance_from_max_gauss) >= 0.00069:
    #     print('Error: Distance_from_max_gauss too large. Plotting results.')
    #     #plot_amplitude_line_with_gaussian(t_line3, amplitude_line3)

    # if Gauss_z_value <= max_amplitude/20:
    #     print(f'Error: Gauss_z_value {Gauss_z_value} below threshold {max_amplitude /20}.')

    # if abs(Freq_from_max_gauss) >= 1300:
    #     print('Error: Freq_from_max_gauss exceeded 1300 Hz. Plotting results.')
    #     #plot_amplitude_line_with_gaussian(t_line3, amplitude_line3)

    # if i >= 500:
    #     print('Warning: Maximum iteration limit (500) reached.')

    # if (LeftRight == 0 and Gauss_x_value > min_previous_x) or (LeftRight == 1 and Gauss_x_value < max_previous_x):
    #     print('Gauss_x_value < or > min_previous_x')

    # if (LeftRight == 0 and Gauss_x_value > limit_Gauss_x_value + 0.00002) or (LeftRight == 1 and Gauss_x_value < limit_Gauss_x_value - 0.00002):
    #     print('Gauss_x_value < or > limit_Gauss_x_value')

    # Reaching the guard means that this side did not converge.  Do not let a
    # partial curve be mistaken for a successfully analysed chirp: propagate
    # failure to process_full_spectrum(), which will discard this chirp.
    if i >= max_iterations:
        return None

    return Seven_points

def process_chunk_spectrum(y_use, sr, time_mid, duration):

  y_chun = Extract_chunk_of_audio(y_use, sr, time_mid)

  # Initial processing

  curve_all = initial_call_trend(y_chun, sr, duration)
  S = np.abs(librosa.stft(y_chun, n_fft=1024, hop_length=100, window='flattop'))
  #curve_all = curve_approx
  # Frequency and time axes
  freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
  times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=100)
  max_value = max(entry[2] for entry in curve_all)
  #print('max_value',max_value)

  #process left side
  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value)
  #print(curve_all)
  #np.savetxt("curve_all.csv", curve_all, delimiter=",")

  #process right side
  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value)

  #plot_spectrogram_with_points_squared(y_chun, sr, curve_all)
  #print(slope_perp)
  #plot_spectrogram_with_points_and_predicted(y_chun, sr, curve_all, a[0])

  return curve_all

#np.savetxt("spectrogram.csv", S, delimiter=",")

## Exemple2

# Transcribing segmented echolocation calls

## Get segmented call position

def find_max_gauss_z(Seven_points, Spectro, freqs, times, sr, LeftRight, max_step=11):
    if Seven_points is None:
        return None

    # Initialize variables
    max_Gauss_z_value = float('-inf')
    max_Distance_from_max_gauss = float('-inf')
    best_i = None

    for i in range(1, max_step):  # i from 1 to 10
        step = 500 * i  # Compute step size

        # Compute extrapolated point and its slope
        NewPointExtrapolated, NewPointExtrapolatedSlope = get_extrapolated_points(
            Seven_points, step_ns=step, LeftRight=LeftRight, nb_point_to_consider=25
        )


        # Sample line from max amplitude
        t_line3, amplitude_line3 = sample_line_from_max_amp_dynamic(
        Spectro, freqs, times, NewPointExtrapolated, sr, slope=NewPointExtrapolatedSlope
        )

        #plot_amplitude_line_with_gaussian(t_line3, amplitude_line3)

        # Compute Gaussian variation and extract Gauss_z_value
        Gaussian_variation_percent, Gauss_z_value, Distance_from_max_gauss, Gaussian_function, Freq_from_max_gauss = calculate_percentage_variation( t_line3, amplitude_line3, slope=NewPointExtrapolatedSlope )

        # Update maximum Gauss_z_value if a new max is found
        if Gauss_z_value > max_Gauss_z_value:
            max_Gauss_z_value = Gauss_z_value
            max_Distance_from_max_gauss = Distance_from_max_gauss
            max_NewPointExtrapolated = NewPointExtrapolated
            max_step=step

    Max_Gauss_coordinate = point_along_line(max_NewPointExtrapolated, NewPointExtrapolatedSlope, max_Distance_from_max_gauss)
    #print(f"NewPointExtrapolated: {max_NewPointExtrapolated}")
    #print(f"NewPointExtrapolatedSlope: {NewPointExtrapolatedSlope}")
    #print(f"Distance_from_max_gauss: {max_Distance_from_max_gauss}")
    #print(f"Max_Gauss_coordinate: {Max_Gauss_coordinate}")
    #print(f"max_Gauss_z_value: {max_Gauss_z_value}")
    #print(f"max_step: {max_step}")


    return Max_Gauss_coordinate

def initial_call_trend_segmented(yt, sr, Max_Gauss_coordinate, step_ns=50):
    """
    Computes the spectrogram of the given audio signal y, finds the maximum amplitude index,
    and analyzes the trend using Gaussian fit error metrics along sampled amplitude lines.

    Returns:
        A NumPy array where each row corresponds to a candidate index and contains:
        - max_time (float): Adjusted time coordinate of the maximum Gaussian fit.
        - max_freq (float): Adjusted frequency coordinate of the maximum Gaussian fit.
        - max_gauss_z (float): Maximum value of the Gaussian fit.
        - percent_variation (float): Percentage variation between the data and the Gaussian fit.
        - max_distance (float): Distance along t_line between the maximum positions.
        - slope (float): Fixed slope value used for the line sampling.
    """
    if Max_Gauss_coordinate is None:
        return None

    # Extract coordinates from Max_Gauss_coordinate
    Gauss_y_value, Gauss_x_value = Max_Gauss_coordinate
    # Compute spectrogram
    Sp = np.abs(librosa.stft(yt, n_fft=1024, hop_length=100, window='flattop'))

    # Frequency and time axes
    freqss = librosa.fft_frequencies(sr=sr, n_fft=1024)
    timess = librosa.frames_to_time(np.arange(Sp.shape[1]), sr=sr, hop_length=100)

    # Find the max amplitude point
    max_freq, max_time = Max_Gauss_coordinate

    step = step_ns/1000000

    # Define candidate indices (time axis adjustments)
    time_offsets = [-3*step,-2*step, -1*step, 0*step, 1*step, 2*step, 3*step]
    candidate_timess = max_time + np.array(time_offsets)
    candidate_freqss = np.full_like(candidate_timess, max_freq)

    # Initialize results array
    results = np.zeros((len(candidate_timess), 9))  # 9 columns for different computed values

    for i, (candidate_time, candidate_freq) in enumerate(zip(candidate_timess, candidate_freqss)):
        # Sample amplitude values along a line through the candidate point
        candidate_point = (candidate_freq, candidate_time)  # Create a tuple (freq, time)

        t_line, amplitude_line = sample_line_from_max_amp_dynamic(Sp, freqss, timess, candidate_point, sr)

        #plot_amplitude_line_with_gaussian(t_line, amplitude_line)

        # Compute Gaussian fit metrics
        percent_variation, max_gauss_z, max_dist_gauss, popt, Freq_from_max_gauss = calculate_percentage_variation(t_line, amplitude_line, slope = 1)

        # Compute the adjusted coordinates using the max distance along the line
        max_gauss_freq, max_gauss_time  = point_along_line(candidate_point, 1, max_dist_gauss)

        # Store results in the array
        if popt is not None and len(popt) >= 3 and None not in (max_gauss_time, max_gauss_freq, max_gauss_z, percent_variation, max_dist_gauss):
            results[i] = [max_gauss_time, max_gauss_freq, max_gauss_z, percent_variation, max_dist_gauss, popt[0], popt[1], popt[2], 1]

    #print(results)
    #plot_initial_trend(Sp, candidate_freqss, candidate_timess, results)

    return results

def extend_trend_left(y_chun, sr, curve_all, S, freqs, times, max_value):
    if curve_all is None:
        return None

    Max_Gauss_coordinate = find_max_gauss_z(curve_all, S, freqs, times, sr, LeftRight=0,max_step=5)
    #print('Max_Gauss_coordinate',Max_Gauss_coordinate)
    #print('001')
    curve_segmented = initial_call_trend_segmented(y_chun, sr, Max_Gauss_coordinate)
    #print('002')
    curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value, Previous_Curve=curve_all)
    #print('003')
    curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value, Previous_Curve=curve_all)
    #print('004')
    return curve_segmented

def extend_trend_right(y_chun, sr, curve_all, S, freqs, times, max_value):
    if curve_all is None:
        return None

    Max_Gauss_coordinate = find_max_gauss_z(curve_all, S, freqs, times, sr, LeftRight=1,max_step=2)

    curve_segmented = initial_call_trend_segmented(y_chun, sr, Max_Gauss_coordinate)

    curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value, Previous_Curve=curve_all)

    curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value, Previous_Curve=curve_all)

    return curve_segmented

def process_spectrum_segmented(y_use, sr, time_mid, duration):

  if time_mid is None:
    return None

  y_chun = Extract_chunk_of_audio(y_use, sr, time_mid)

  # Initial processing
  curve_approx = initial_call_trend(y_chun, sr, duration)
  S = np.abs(librosa.stft(y_chun, n_fft=1024, hop_length=100, window='flattop'))
  curve_all = curve_approx
  # Frequency and time axes
  freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
  times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=100)
  max_value = max(entry[2] for entry in curve_approx)

  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value)
  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value)

  #plot_spectrogram_with_points2(y_chun, sr, curve_all)
  #print('----------------------------------------------------------------')
  #Process extension1
  Max_Gauss_coordinate = find_max_gauss_z(curve_all, S, freqs, times, sr, LeftRight=0,max_step=11)
  #print('Max_Gauss_coordinate')
  #print(Max_Gauss_coordinate)
  curve_segmented = initial_call_trend_segmented(y_chun, sr, Max_Gauss_coordinate)
  curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value, Previous_Curve=curve_all)
  curve_segmented = process_side(curve_segmented, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value, Previous_Curve=curve_all)


  plot_spectrogram_with_points2(y_chun, sr, curve_segmented)

  return curve_all

def validate_slope(Trend_result, nb_points=25, LeftRight=0):
    """
    Computes the slope of the first or last `nb_points` in `Trend_result` using linear regression.

    Parameters:
    - Trend_result: List of points, where each point is a list or tuple [x, y, ...].
    - nb_points: Number of points to consider for slope calculation (default: 25).
    - LeftRight: Direction of processing (0 for first `nb_points`, 1 for last `nb_points`).

    Returns:
    - Slope of the best-fit line for the selected points, or None if insufficient points.
    """
    if (len(Trend_result) < 2 or Trend_result is None):
        return None  # Not enough points to calculate a slope

    # Select the first or last `nb_points` based on LeftRight
    points_to_consider = (Trend_result[:nb_points] if LeftRight == 0 else Trend_result[-nb_points:])

    # Extract x and y values
    x_values = np.array([point[0] for point in points_to_consider])
    y_values = np.array([point[1] for point in points_to_consider])

    if len(x_values) < 2:
        return None  # Need at least two points to calculate slope

    # Perform linear regression to get the slope
    line, _, _ = fit_polynomial(np.column_stack((x_values, y_values)), degree=1)
    if line is None:
        return None
    slope = line.c[0]

    # Check if the slope is in the desired range (-50 kHz/ms to 0)
    if -50000000 <= slope <= 0:
        return 1
    else:
        return 0

def get_max_frequency(Trend_result):
    """
    Returns the maximum frequency value from Trend_result.

    Parameters:
    - Trend_result: List of points where each point is a list or tuple with frequency at index 1.

    Returns:
    - Maximum frequency value, or None if Trend_result is empty.
    """

    return max(point[1] for point in Trend_result)

def get_min_frequency(Trend_result):
    """
    Returns the maximum frequency value from Trend_result.

    Parameters:
    - Trend_result: List of points where each point is a list or tuple with frequency at index 1.

    Returns:
    - Maximum frequency value, or None if Trend_result is empty.
    """

    return min(point[1] for point in Trend_result)

def get_number_of_points(Trend_result):
    """
    Returns the number of points in Trend_result.

    Parameters:
    - Trend_result: List or NumPy array of points, where each point has multiple parameters.

    Returns:
    - Number of points (rows) in Trend_result.
    """
    if Trend_result is None:
        return 0  # If None, return 0

    if isinstance(Trend_result, np.ndarray):
        return Trend_result.shape[0]  # NumPy array: return number of rows

    return len(Trend_result)  # If it's a list, return its length

def calculate_angle_between_trends(Trend_result1, Trend_result2, nb_points=25, scale_factor=3730000):
    """
    Computes the angle between two trend lines.

    The first line is based on the last `nb_points` from `Trend_result1`,
    and the second line is based on the first `nb_points` from `Trend_result2`.

    Parameters:
    - Trend_result1: List of points defining the first trend, where each point is [x, y, ...].
    - Trend_result2: List of points defining the second trend, where each point is [x, y, ...].
    - nb_points: Number of points to consider for each trend (default: 25).
    - scale_factor: A scaling factor applied to slopes (default: 3730000).

    Returns:
    - Angle in degrees between the two trend lines.
    - Returns None if there are insufficient points.
    """
    # Ensure both lists have at least two points
    if Trend_result1 is None or Trend_result2 is None or len(Trend_result1) < 2 or len(Trend_result2) < 2:
        return None # Not enough points to calculate an angle

    # Extract the last `nb_points` from Trend_result1
    points1 = Trend_result1[-nb_points:]
    x1 = np.array([p[0] for p in points1])
    y1 = np.array([p[1] for p in points1])

    # Extract the first `nb_points` from Trend_result2
    points2 = Trend_result2[:nb_points]
    x2 = np.array([p[0] for p in points2])
    y2 = np.array([p[1] for p in points2])

    # Check if we have enough points for regression
    if len(x1) < 2 or len(x2) < 2:
        return None  # Need at least two points per line

    # Fit linear regressions while rejecting degenerate/duplicate time axes.
    line1, _, _ = fit_polynomial(np.column_stack((x1, y1)), degree=1)
    line2, _, _ = fit_polynomial(np.column_stack((x2, y2)), degree=1)
    if line1 is None or line2 is None:
        return None

    # Compute slopes
    slope1 = line1.c[0] / scale_factor
    slope2 = line2.c[0] / scale_factor

    # Compute the angle between the two slopes
    angle_rad = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
    angle_deg = np.degrees(angle_rad)

    return abs(angle_deg)

def concatenate_trends(trend_result1, trend_result2):
    """
    Concatenates two trend result arrays while ensuring they maintain their structure.
    Assumes both arrays have the same number of columns and sorts them in ascending order based on time.
    """
    concatenated = np.vstack((trend_result1, trend_result2))
    return concatenated  # Assuming the first column represents time

def process_full_spectrum(y_use, sr, time_mid, duration):

  y_chun = Extract_chunk_of_audio(y_use, sr, time_mid)

  # Initial processing
  curve_all = initial_call_trend(y_chun, sr, duration)

  if curve_all is None or len(curve_all) == 0:
    return None

  max_value = max(entry[2] for entry in curve_all)

  if max_value < 0.7:
    #print('No call found')
    return None

  S = np.abs(librosa.stft(y_chun, n_fft=1024, hop_length=100, window='flattop'))
  # Frequency and time axes
  freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
  times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=100)

  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=0, max_amplitude=max_value)

  if curve_all is None:
    return None

  curve_all = process_side(curve_all, S, freqs, times, sr, LeftRight=1, max_amplitude=max_value)

  if curve_all is None:
    return None



  #plot_spectrogram_with_points2(y_chun, sr, curve_all)
  #print('extend to left of the initial trend')

  # Initialize curve_segmented before the loop
  max_curve_segmented = 0

  curve_segmented = extend_trend_left(y_chun, sr, curve_all, S, freqs, times, max_value)

  if curve_segmented is not None:
      max_curve_segmented = max(entry[2] for entry in curve_segmented)
      while (calculate_angle_between_trends(curve_all, curve_segmented)<70 and get_number_of_points(curve_segmented)>11 and (max_curve_segmented>(max_value/20))):
        #print('curve_all nb points before concat 1',get_number_of_points(curve_all))
        curve_all=concatenate_trends(curve_segmented,curve_all)
        #pdb.set_trace()
        #print('curve_all nb points 2',get_number_of_points(curve_all))
        curve_segmented = extend_trend_left(y_chun, sr, curve_all, S, freqs, times, max_value)
        if curve_segmented is None:
          break
        max_curve_segmented = max(entry[2] for entry in curve_segmented)
        #pdb.set_trace()

  #print('curve_all nb points beginin',get_number_of_points(curve_all))
  #print('curve_segmented nb points',get_number_of_points(curve_segmented))
  #print('calculate_angle_between_trends',calculate_angle_between_trends(curve_all, curve_segmented))
  #print('get_number_of_points',get_number_of_points(curve_segmented))


  #plot_spectrogram_with_points2(y_chun, sr, curve_segmented)
  #print('calculate_angle_between_trends(curve_all, curve_segmented)<50',calculate_angle_between_trends(curve_all, curve_segmented))
  #print('get_number_of_points(curve_segmented)>11',get_number_of_points(curve_segmented))
  #print('(max_curve_segmented>(max_value/10))',(max_curve_segmented>(max_value/10)))

  curve_segmented = extend_trend_right(y_chun, sr, curve_all, S, freqs, times, max_value)

  if curve_segmented is not None:
      max_curve_segmented = max(entry[2] for entry in curve_segmented)
      while (calculate_angle_between_trends(curve_all, curve_segmented)<50 and get_number_of_points(curve_segmented)>11 and validate_slope(curve_segmented, nb_points=25, LeftRight=0) and  get_max_frequency(curve_segmented)<(get_min_frequency(curve_all)+500) and max_curve_segmented>max_value/10):
          curve_all=concatenate_trends(curve_all,curve_segmented)
          curve_segmented = extend_trend_right(y_chun, sr, curve_all, S, freqs, times, max_value)
          if curve_segmented is None:
              break
          max_curve_segmented = max(entry[2] for entry in curve_segmented)



  #pdb.set_trace()
  #print('while right')


  if curve_all is None:
    return None

  curve_all = interpolate_trend_results(curve_all)

  #plot_spectrogram_with_points2(y_chun, sr, curve_all)
  curve_all = fit_spline_with_smoothness(curve_all)
  #plot('sample_curve_equally')
  curve_all = sample_curve_equally(curve_all)
  #call_fitted = downsample_douglas_peucker(curve_all,epsilon=0.00002)

  #print('plot curve modelised')
  #plot_spectrogram_with_points2(y_chun, sr, curve_all)
  #plot_spectrogram_with_points2(y_chun, sr, call_fitted)

  return curve_all

def interpolate_trend_results(trend_result, max_gap=0.00002):
    """
    Inserts points between gaps in trend_result using linear interpolation
    so that no gap exceeds max_gap (in ms).
    """
    times = trend_result[:, 0]
    values = trend_result[:, 1:]

    new_times = [times[0]]
    new_values = [values[0]]

    for i in range(1, len(times)):
        time_diff = times[i] - times[i-1]
        if time_diff > max_gap:
            num_points = int(np.ceil(time_diff / max_gap))
            interpolated_times = np.linspace(times[i-1], times[i], num_points, endpoint=False)[1:]
            interpolated_values = np.linspace(values[i-1], values[i], num_points, endpoint=False)[1:]
            new_times.extend(interpolated_times)
            new_values.extend(interpolated_values)

        new_times.append(times[i])
        new_values.append(values[i])

    return np.column_stack((new_times, new_values))

from scipy.interpolate import UnivariateSpline

def fit_spline_with_smoothness(trend_results, num_points=100, smoothness=1):
    times = trend_results[:, 0]
    y_values = trend_results[:, 1]
    z_values = trend_results[:, 2]

    # Define a parametric variable t
    t = np.linspace(0, 1, len(times))

    # Fit splines with smoothness parameter `s`
    spline_x = UnivariateSpline(t, times, s=smoothness*0.00000001)
    spline_y = UnivariateSpline(t, y_values, s=smoothness*1000000)
    spline_z = UnivariateSpline(t, z_values, s=smoothness*0.1)

    # Fit splines with smoothness=0
    spline_x_b = UnivariateSpline(t, times, s=0)
    spline_y_b = UnivariateSpline(t, y_values, s=0)
    spline_z_b = UnivariateSpline(t, z_values, s=0)

    # Sample exactly `num_points` equally spaced
    sampled_t = np.linspace(0, 1, num_points)
    sampled_x = spline_x(sampled_t)
    sampled_y = spline_y(sampled_t)
    sampled_z = spline_z(sampled_t)
    sampled_x_b = spline_x_b(sampled_t)
    sampled_y_b = spline_y_b(sampled_t)
    sampled_z_b = spline_z_b(sampled_t)

    #plot_spline_x_y(sampled_t, sampled_x, sampled_y, sampled_z, sampled_x_b, sampled_y_b, sampled_z_b)

    return np.column_stack((sampled_x, sampled_y, sampled_z))

from shapely.geometry import LineString

def downsample_douglas_peucker(trend_results, epsilon=0.01):
    """
    Downsamples trend_results using the Douglas-Peucker algorithm while preserving z-values.

    Parameters:
    - trend_results: 2D NumPy array with columns [time, y, z].
    - epsilon: Tolerance for simplification (higher = more aggressive downsampling).

    Returns:
    - A 2D NumPy array with columns [time, y, z] after downsampling.
    """
    # Extract (x, y) for simplification
    line = LineString(trend_results[:, :2])
    simplified_xy = np.array(line.simplify(epsilon, preserve_topology=False).coords)

    # Match z-values based on the nearest original points
    indices = [np.argmin(np.linalg.norm(trend_results[:, :2] - point, axis=1)) for point in simplified_xy]
    simplified_z = trend_results[indices, 2]

    return np.column_stack((simplified_xy, simplified_z))

def sample_curve_equally(curve_all, num_points=100):
    """
    Extracts `num_points` evenly spaced along the time axis from curve_all.

    Parameters:
    - curve_all: NumPy array with shape (N, 2), where:
      - curve_all[:, 0] -> Time values
      - curve_all[:, 1] -> Frequency values
    - num_points: Number of points to sample (default: 100)

    Returns:
    - Sampled NumPy array with shape (num_points, 2)
    """
    times = curve_all[:, 0]
    freqs = curve_all[:, 1]
    zval = curve_all[:, 2]

    # Generate 100 evenly spaced time points
    sampled_times = np.linspace(times.min(), times.max(), num_points)

    # Interpolate frequencies at these times
    sampled_freqs = np.interp(sampled_times, times, freqs)
    sampled_z = np.interp(sampled_times, times, zval)

    return np.column_stack((sampled_times, sampled_freqs, sampled_z))

# Distinguishing echolocation call regions

## Extract signal

def extract_call_from_audio(y, sr, t_start, t_end):
    """
    Extracts an audio chunk from the audio data `y` based on the specified start and end times.

    Parameters:
    - y (numpy.ndarray): Audio time series (1D NumPy array).
    - sr (int): Sampling rate of the audio.
    - t_start (float): Start time of the chunk in seconds.
    - t_end (float): End time of the chunk in seconds.

    Returns:
    - numpy.ndarray: The extracted audio chunk.
    """

    # Convert time to sample indices
    start_sample = max(0, int(t_start * sr))
    end_sample = min(len(y), int(t_end * sr))

    # Extract the audio chunk
    truncated_y = y[start_sample:end_sample]

    #plot_spectrogram_plotly(truncated_y, sr,n_fft=1024, hop_length=100)

    return truncated_y

from scipy.interpolate import interp1d

def central_difference(curve_all, region_start, region_end):
    """
    Calculate the central difference of a given region.

    :param region_signal: The signal in the region.
    :param region_start: The start index of the region.
    :param region_end: The end index of the region.
    :return: The central difference of the region.
    """
    times = curve_all[:, 0]  # Time values
    freqs = curve_all[:, 1]  # Frequency values

    #print('region_start[0]',region_start[0])
    #print('region_end[0]',region_end[0])

    middle_point = (region_start[0] + region_end[0]) / 2
    #print('middle_point',middle_point)
    interpolator = interp1d(times, freqs, kind='linear', fill_value="extrapolate")
    frequency_middle_point = interpolator(middle_point)
    #print('frequency_middle_point',frequency_middle_point)

    prev_x, next_x = region_start[0], region_end[0]
    prev_y, next_y = region_start[1], region_end[1]

    # print('prev_x',prev_x)
    # print('middle_point',middle_point)
    # print('next_x',next_x)
    # print('prev_y',prev_y)
    # print('frequency_middle_point',frequency_middle_point)
    # print('next_y',next_y)
    if ((middle_point == prev_x) or (next_x == middle_point)):
      return 0

    # Compute slope at each side
    slope1 = ((frequency_middle_point - prev_y) / (middle_point - prev_x))
    slope1deg = np.degrees(np.arctan(slope1/3730000))
    slope2 = ((next_y - frequency_middle_point) / (next_x - middle_point))
    slope2deg = np.degrees(np.arctan(slope2/3730000))
    # print('slope1',slope1)
    # print('slope2',slope2)

    # print('-----------------')



    return slope2deg-slope1deg

import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks

def compute_amplitude_bandwidths(signal, sampling_rate):
    """
    Computes the amplitude bandwidths at 80%, 50%, and 30% of the maximum amplitude.

    :param signal: 1D array of the signal.
    :param sampling_rate: Sampling rate in Hz.
    :return: Tuple (80%AmpBand, 50%AmpBand, 30%AmpBand)
    """
    N = len(signal)
    spectrum = np.abs(fft(signal))[:N//2]  # Compute FFT magnitude
    freq_axis = np.fft.fftfreq(N, d=1/sampling_rate)[:N//2]  # Positive frequencies

    max_amp = np.max(spectrum)  # Maximum amplitude in the power spectrum

    def find_bandwidth(threshold):
        """Finds the frequency range where amplitude is above 'threshold' % of max_amp."""
        limit = threshold * max_amp
        valid_freqs = freq_axis[spectrum >= limit]
        if len(valid_freqs) == 0:
            return 0  # No valid range found
        return valid_freqs[-1] - valid_freqs[0]  # High - Low frequency

    amp_80_band = find_bandwidth(0.8)
    amp_50_band = find_bandwidth(0.5)
    amp_30_band = find_bandwidth(0.3)

    return amp_80_band, amp_50_band, amp_30_band

import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks

def extract_max_tone_amplitude(segment, sampling_rate):
    """
    Extracts the maximum amplitude of the dominant frequency in a given segment.

    :param segment: 1D array of the signal segment.
    :param sampling_rate: Sampling rate in Hz.
    :return: Maximum amplitude of the dominant frequency.
    """
    N = len(segment)
    spectrum = np.abs(fft(segment))[:N//2]  # Compute FFT magnitude
    freq_axis = np.fft.fftfreq(N, d=1/sampling_rate)[:N//2]  # Positive frequencies

    # Find the dominant frequency peak
    peaks, properties = find_peaks(spectrum, height=np.max(spectrum) * 0.1)  # Detect peaks
    if len(peaks) == 0:
        return 0  # No dominant frequency detected

    max_peak_index = peaks[np.argmax(properties["peak_heights"])]
    max_amplitude = spectrum[max_peak_index]

    return max_amplitude

def compute_quartile_amplitudes(raw_signal, sampling_rate):
    """
    Computes the Q1 to Q4 quartile amplitudes of a signal.

    :param raw_signal: 1D array of the raw signal.
    :param sampling_rate: Sampling rate in Hz.
    :return: Tuple of (Q1, Q2, Q3, Q4).
    """
    signal_length = len(raw_signal)
    quarter_size = signal_length // 4  # Divide the signal into 4 equal parts

    # Extract max tone amplitude for each quarter
    Q1 = extract_max_tone_amplitude(raw_signal[:quarter_size], sampling_rate)
    Q2 = extract_max_tone_amplitude(raw_signal[quarter_size:2*quarter_size], sampling_rate)
    Q3 = extract_max_tone_amplitude(raw_signal[2*quarter_size:3*quarter_size], sampling_rate)
    Q4 = extract_max_tone_amplitude(raw_signal[3*quarter_size:], sampling_rate)

    return Q1, Q2, Q3, Q4

import numpy as np
from scipy.fftpack import fft
from scipy.signal import find_peaks

def compute_thd(signal, sampling_rate):
    """
    Computes Total Harmonic Distortion (THD) of a signal.

    :param signal: 1024-sample segment of the raw signal.
    :param sampling_rate: Sampling rate in Hz.
    :return: THD percentage.
    """
    N = len(signal)
    #print('N',N)
    freq_axis = np.fft.fftfreq(N, d=1/sampling_rate)[:N//2]  # Positive frequencies
    spectrum = np.abs(fft(signal))[:N//2]  # Compute FFT magnitude

    # Find the fundamental frequency (highest peak in spectrum)
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1)  # Detect peaks
    if len(peaks) == 0:
        return 0  # No fundamental frequency detected

    fundamental_index = peaks[0]
    fundamental_amplitude = spectrum[fundamental_index]
    fundamental_freq = freq_axis[fundamental_index]

    #print('fundamental_amplitude',fundamental_amplitude)
    #print('fundamental_freq',fundamental_freq)

    # Identify harmonics (multiples of fundamental frequency)
    harmonic_amplitudes = []
    for i in range(2, 6):  # Check up to the 5th harmonic
        harmonic_freq = i * fundamental_freq
        harmonic_index = np.argmin(np.abs(freq_axis - harmonic_freq))  # Closest match
        harmonic_amplitudes.append(spectrum[harmonic_index])

    # Calculate THD
    thd_value = (np.sqrt(np.sum(np.square(harmonic_amplitudes))) / fundamental_amplitude) * 100
    #print('thd_value',thd_value)
    return thd_value

def get_max_thd(raw_signal, sampling_rate):
    """
    Computes the maximum THD from three 1024-sample segments taken at 1/4, 1/2, and 3/4 of the call duration.

    :param raw_signal: Full raw signal array.
    :param sampling_rate: Sampling rate in Hz.
    :return: Maximum THD value.
    """
    signal_length = len(raw_signal)
    #print('signal_length',signal_length)
    segment_size = 1024

    # Ensure segment indices are within valid bounds
    start_indices = [
        max(0, min(signal_length - segment_size, signal_length // 4 - segment_size // 2)),
        max(0, min(signal_length - segment_size, signal_length // 2 - segment_size // 2)),
        max(0, min(signal_length - segment_size, (3 * signal_length) // 4 - segment_size // 2))
    ]

    # Compute THD for each segment
    thd_values = []
    for start in start_indices:
        end = min(signal_length, start + segment_size)  # Ensure end is within bounds
        segment = raw_signal[start:end]

        if len(segment) == segment_size:  # Ensure we get the full segment
            thd_values.append(compute_thd(segment, sampling_rate))

    # Return the maximum THD found
    return max(thd_values) if thd_values else None  # Handle empty case

def measure_duration(curve_all, threshold_ratio):
    """
    Measures the total duration where amplitude is above threshold_ratio * max amplitude.

    :param curve_all: 2D NumPy array where column 0 is time and column 1 is amplitude.
    :param threshold_ratio: Fraction of max amplitude (e.g., 0.75, 0.50, 0.25).
    :return: Total duration where amplitude is above threshold.
    """
    time_values = curve_all[:, 0]
    amplitude_values = curve_all[:, 2]

    # Compute maximum amplitude
    max_amplitude = np.max(amplitude_values)

    # Compute threshold value
    threshold_value = (threshold_ratio * max_amplitude) / 100
    #print('threshold_value',threshold_value)

    # Create boolean mask: True where amplitude is above the threshold
    above_threshold = amplitude_values >= threshold_value
    #print('above_threshold',above_threshold)

    # Identify regions where the signal is above the threshold
    durations = []
    in_segment = False
    start_time = None

    for i in range(len(time_values)):
        if above_threshold[i]:  # Signal is above threshold
            if not in_segment:  # Start of a new segment
                start_time = time_values[i]
                in_segment = True
        else:  # Signal is below threshold
            if in_segment:  # End of a segment
                durations.append(time_values[i] - start_time)
                in_segment = False

    # If the last segment reaches the end of the data, close it
    if in_segment:
        durations.append(time_values[-1] - start_time)


    # Compute total duration above threshold
    return sum(durations)

def compute_central_difference(y_values, x_values, window_size):
    """Compute central difference curvature over a given window size."""
    curvature = np.full_like(y_values, np.nan)
    angle_deg = np.full_like(y_values, np.nan)
    window_size = int(window_size/2)

    for i in range(window_size, len(y_values) - window_size):
        prev_x, next_x = x_values[i - window_size], x_values[i + window_size]
        prev_y, next_y = y_values[i - window_size], y_values[i + window_size]

        # Compute slope at each side
        slope1 = ((y_values[i] - prev_y) / (x_values[i] - prev_x))
        slope1deg = np.degrees(np.arctan(slope1/3730000))
        slope2 = ((next_y - y_values[i]) / (next_x - x_values[i]))
        slope2deg = np.degrees(np.arctan(slope2/3730000))


        # Compute central difference curvature
        curvature[i] = slope2 - slope1
        angle_deg[i] = slope2deg - slope1deg

    return curvature, angle_deg


def find_knee_and_characteristic_points(curve_all):
    """
    Identify:
    - Upper knee (max curvature using slope from both small and large windows)
    - Lower knee (max curvature using angle from both small and large windows)
    - Characteristic frequency (min curvature using angle from large window)

    Parameters:
    - curve_all: NumPy array with shape (N, 2), where:
      - curve_all[:, 0] -> Time values
      - curve_all[:, 1] -> Frequency values
    """

    times = curve_all[:, 0]
    freqs = curve_all[:, 1]

    duration = times[-1] - times[0]

    small_window = max(1, int(len(times) * 0.1))  # 1/10th of the duration
    large_window = max(1, int(len(times) * 0.2))  # 1/5th of the duration

    # Compute slope curvature
    slope_curvature_small, angle_curvature_small = compute_central_difference(freqs, times, small_window)
    slope_curvature_large, angle_curvature_large = compute_central_difference(freqs, times, large_window)

    # Find key points based on max curvature
    #upper_knee_index = np.nanargmax(np.maximum(slope_curvature_small, slope_curvature_large))
    #lower_knee_index = np.nanargmax(np.maximum(angle_curvature_small, angle_curvature_large))
    #characteristic_index = np.nanargmin(angle_curvature_large)  # Min angle curvature (large window)
    upper_knee_index = np.nanargmax(np.fmax(slope_curvature_small, slope_curvature_large))
    lower_knee_index = np.nanargmax(np.fmax(angle_curvature_small, angle_curvature_large))
    characteristic_index = np.nanargmin(np.fmin(angle_curvature_small, angle_curvature_large))

    Maximum_central_difference = np.nanmax(np.fmax(angle_curvature_small, angle_curvature_large))

    # # Get the indices of the max values in each array
    # print(slope_curvature_small)
    # index_slope_small = np.argmax(slope_curvature_small)
    # print('index_slope_small',index_slope_small)
    # index_slope_large = np.argmax(slope_curvature_large)
    # index_angle_small = np.argmax(angle_curvature_small)
    # index_angle_large = np.argmax(angle_curvature_large)
    # index_angle_small_min = np.argmin(angle_curvature_small)
    # index_angle_large_min = np.argmin(angle_curvature_large)

    # # Compare the max values and choose the corresponding index
    # upper_knee_index = index_slope_small if slope_curvature_small[index_slope_small] > slope_curvature_large[index_slope_large] else index_slope_large
    # lower_knee_index = index_angle_small if angle_curvature_small[index_angle_small] > angle_curvature_large[index_angle_large] else index_angle_large
    # characteristic_index = index_angle_small_min if angle_curvature_small[index_angle_small_min] < angle_curvature_large[index_angle_large_min] else index_angle_large_min


    #print("Upper Knee Index:", upper_knee_index)
    #print("Lower Knee Index:", lower_knee_index)
    #print("Characteristic Index:", characteristic_index)

    # Plot curvature graphs
    #plot_curvature(times, slope_curvature_small, slope_curvature_large, angle_curvature_small, angle_curvature_large)

    return {
        "Upper_Knee": (times[upper_knee_index], round(freqs[upper_knee_index], 1)),
        "Lower_Knee": (times[lower_knee_index], round(freqs[lower_knee_index], 1)),
        "Characteristic_Frequency": (times[characteristic_index], round(freqs[characteristic_index], 1)),
        "Maximum_central_difference": round(Maximum_central_difference,1)
    }

def amplitude_to_db(audio_signal, bit_depth):
    """ Convert an audio signal to dB scale using bit depth. """
    max_amplitude = (2 ** (bit_depth - 1)) - 1  # e.g., 32767 for 16-bit audio
    return 20 * np.log10(np.maximum(np.abs(audio_signal), 1e-12) / max_amplitude)

def get_slope(y2, y1, x2, x1):
    if (x2 == x1):
        return 0
    return (y2 - y1) / (x2 - x1)

from dataclasses import dataclass

@dataclass
class Call_parameter:
    t_start_in_file : float
    t_start : float
    t_end : float
    Duration : float
    Level_dB : float
    Level : float
    Maximum_frequency : float
    Minimum_frequency : float
    Start_frequency : float
    End_frequency : float
    Bandwith : float
    Frequency_max_amplitude : float
    Time_to_maximum : float
    Upper_knee_frequency : float
    Lower_knee_frequency : float
    Characteristic_frequency : float
    F_center : float
    Slope : float
    Time_to_lower_knee_percent : float
    Upper_slope : float
    Knee_Slope : float
    Body_slope : float
    Tail_slope : float
    Percent_75_amplitude : float
    Percent_50_amplitude : float
    Percent_25_amplitude : float
    THD : float
    Q1 : float
    Q2 : float
    Q3 : float
    Q4 : float
    Percent_80_amplitude_bandwith : float
    Percent_50_amplitude_bandwith : float
    Percent_30_amplitude_bandwith : float
    Upper_central_difference : float
    Knee_central_difference : float
    Body_central_difference : float
    Tail_central_difference : float
    Maximum_central_difference : float
    Upper_length : float
    Knee_length : float
    Body_length : float
    Tail_length : float
    #Inter_pulse_interval : float
    #Standard_deviation_inter_pulse_interval : float
    #Standard_deviation_characteristic_frequency : float

def extract_curve_features(curve_all,signal_full,sr,tim):

    if curve_all is None:
        return None
    # Extract time and frequency values
    time_values = curve_all[:, 0]
    freq_values = curve_all[:, 1]

    knee_points = find_knee_and_characteristic_points(curve_all)

    # Accessing time and frequency for Upper Knee
    upper_knee_time, upper_knee_frequency = knee_points["Upper_Knee"]

    # Accessing time and frequency for Lower Knee
    lower_knee_time, lower_knee_frequency = knee_points["Lower_Knee"]

    # Accessing time and frequency for Characteristic Frequency
    characteristic_time, characteristic_frequency = knee_points["Characteristic_Frequency"]

    # Compute start and end times
    t_start, f_start = curve_all[0,0], curve_all[0,1]
    t_end, f_end = curve_all[-1,0], curve_all[-1,1]

    bandwith = np.max(curve_all[:, 1]) - np.min(curve_all[:, 1])

    if characteristic_time<lower_knee_time or characteristic_time<upper_knee_time or lower_knee_time<upper_knee_time or 1000*(t_end-t_start)<1 or bandwith<10000:
        return None

    max_cd = knee_points["Maximum_central_difference"]

    # Find the index of the maximum value in column 2 (curve values)
    max_index = np.argmax(curve_all[:, 2])
    # Get the corresponding time from column 0
    time_at_max = curve_all[max_index, 0]
    # Get the starting time
    start_time = curve_all[0, 0]
    # Compute start and end times
    t_start, f_start = curve_all[0,0], curve_all[0,1]
    t_end, f_end = curve_all[-1,0], curve_all[-1,1]

    # Compute half-duration time
    t_half = t_start + (curve_all[-1, 0] - curve_all[0, 0]) / 2

    signal = extract_call_from_audio(signal_full, sr, t_start, t_end)

    thd = get_max_thd(signal, sr)

    Q1, Q2, Q3, Q4 = compute_quartile_amplitudes(signal, sr)

    amp_80, amp_50, amp_30 = compute_amplitude_bandwidths(signal, sr)


    # Calculate central differences for each region
    up_cd = central_difference(curve_all, curve_all[0], knee_points["Upper_Knee"])
    knee_cd = central_difference(curve_all, knee_points["Upper_Knee"], knee_points["Lower_Knee"])
    body_cd = central_difference(curve_all, knee_points["Lower_Knee"], knee_points["Characteristic_Frequency"])
    tail_cd = central_difference(curve_all, knee_points["Characteristic_Frequency"], curve_all[-1])

    #plot_spectrogram_with_knee(signal_full, sr, curve_all, knee_points, n_fft=1024, hop_length=100, window='flattop')


    return Call_parameter(
        t_start_in_file=(tim+t_start-(15/2)/1000), #window are 15ms long and tim is in the middle of this window
        t_start=t_start,
        t_end=t_end,
        Duration=round(1000*(t_end-t_start),3),
        Level_dB=round(amplitude_to_db(curve_all[max_index, 2],16), 1),
        Level=round(curve_all[max_index, 2], 1),
        Maximum_frequency=round(np.max(curve_all[:, 1]), 1),
        Minimum_frequency=round(np.min(curve_all[:, 1]), 1),
        Start_frequency=round(curve_all[0, 1], 1),
        End_frequency=round(curve_all[-1, 1], 1),
        Bandwith=round(bandwith, 1),
        Frequency_max_amplitude=round(curve_all[max_index, 1], 1),
        Time_to_maximum = round(1000*(time_at_max-start_time),3),
        Upper_knee_frequency = upper_knee_frequency,
        Lower_knee_frequency = lower_knee_frequency,
        Characteristic_frequency = characteristic_frequency,
        F_center=round(np.interp(t_half, time_values, freq_values),1),
        Slope=round((f_end - f_start) / (t_end - t_start),1),
        Time_to_lower_knee_percent=round((lower_knee_time-start_time)/(t_end-t_start)*100,1),
        Upper_slope = round(get_slope(upper_knee_frequency, f_start, upper_knee_time, t_start),1),
        Knee_Slope = round(get_slope(lower_knee_frequency, upper_knee_frequency, lower_knee_time, upper_knee_time),1),
        Body_slope = round(get_slope(characteristic_frequency, lower_knee_frequency, characteristic_time, lower_knee_time),1),
        Tail_slope = round(get_slope(characteristic_frequency, f_end, characteristic_time, t_end),1),
        Percent_75_amplitude = round(measure_duration(curve_all,75)*100/(t_end-t_start),1),
        Percent_50_amplitude = round(measure_duration(curve_all,50)*100/(t_end-t_start),1),
        Percent_25_amplitude = round(measure_duration(curve_all,25)*100/(t_end-t_start),1),
        THD=round(thd or 0,1),
        Q1=round(Q1,1),
        Q2=round(Q2,1),
        Q3=round(Q3,1),
        Q4=round(Q4,1),
        Percent_80_amplitude_bandwith=round(amp_80,1),
        Percent_50_amplitude_bandwith=round(amp_50,1),
        Percent_30_amplitude_bandwith=round(amp_30,1),
        Upper_central_difference=round(up_cd,1),
        Knee_central_difference=round(knee_cd,1),
        Body_central_difference=round(body_cd,1),
        Tail_central_difference=round(tail_cd,1),
        Maximum_central_difference=max_cd,
        Upper_length=round(1000*(upper_knee_time - t_start),3),
        Knee_length=round(1000*(lower_knee_time - upper_knee_time),3),
        Body_length=round(1000*(characteristic_time - lower_knee_time),3),
        Tail_length=round(1000*(t_end - characteristic_time),3))

from pprint import pprint

def process_wav_file(file_path, *, detector: str = "snr_blob"):
    """Process one WAV file and return a list of extracted chirp feature dicts.

    detector:
      - "snr_blob" (default): SNR-thresholded blob detector (from untitled13.py idea)
      - "zcr": legacy zero-crossing based midpoint detector
    """
    results_table: List[Dict[str, Any]] = []

    y, sr = librosa.load(file_path, sr=None)
    y_filtered = high_pass_filter(y, sr)
    y_use = y_filtered

    if detector == "snr_blob":
        candidates = detect_candidates_snr_blobs(
            y_use,
            sr,
            snr_threshold_db=10.0,
            percentile_q=96.0,
            fmin=20000,
            fmax=150000,
            n_fft=512,
            hop=128,
            min_blob_size=10,
            min_blob_height_hz=5000.0,
            max_blob_slope_hz_per_ms=-2000.0,
            echo_suppression_window_ms=10.0,
        )
        # convert to (time_mid, duration) like legacy
        mid_time = [(c["time_mid"], c["duration"]) for c in candidates]

    elif detector == "zcr":
        zero_crossing_freq, zero_crossing_times = compute_zero_crossing_frequency(
            y_use, sr, amplitude_threshold=5
        )
        stable_regions, _times = detect_stable_frequency_regions(zero_crossing_freq, zero_crossing_times)
        mid_time = compute_stable_region_midpoints(stable_regions, zero_crossing_times)

    else:
        raise ValueError("detector must be 'snr_blob' or 'zcr'")

    for chirp_index, (tim, dur) in enumerate(mid_time, start=1):
        # keep a sensible analysis window even if the blob bbox is tiny
        dur = float(np.clip(dur, 0.005, 0.08))

        try:
            y_chunk = Extract_chunk_of_audio(y_use, sr, tim)
            c = process_full_spectrum(y_use, sr, time_mid=tim, duration=dur)

            # None is the normal failure signal for an analysis that could not
            # produce a valid complete curve.  Skip it and try the next chirp.
            if c is None:
                print(
                    f"  Skipping chirp {chirp_index} at {tim:.6f} s: "
                    "analysis did not converge or returned no valid curve."
                )
                continue

            result = extract_curve_features(c, y_chunk, sr, tim)
            if result is not None:
                results_table.append(result)
            else:
                print(
                    f"  Skipping chirp {chirp_index} at {tim:.6f} s: "
                    "feature extraction returned no result."
                )

        except Exception as exc:
            # A malformed candidate must not stop the remaining chirps in this
            # WAV. KeyboardInterrupt/SystemExit are intentionally not caught.
            print(
                f"  Skipping chirp {chirp_index} at {tim:.6f} s "
                f"after {type(exc).__name__}: {exc}"
            )
            continue

    return results_table

def parse_guano_metadata(metadata):
    # Extracting specific metadata values
    version = metadata.get('GUANO|Version', 'Unknown')
    length = metadata.get('Length', 'Unknown')
    location = metadata.get('Loc Position', 'Unknown')
    latitude = location[0] if isinstance(location, tuple) else 'Unknown'
    longitude = location[1] if isinstance(location, tuple) else 'Unknown'
    elevation = metadata.get('Loc Elevation', 'Unknown')
    make = metadata.get('Make', 'Unknown')
    model = metadata.get('Model', 'Unknown')
    original_filename = metadata.get('Original Filename', 'Unknown')
    bit_depth = metadata.get('Bit Depth', 'Unknown')
    samplerate = metadata.get('Samplerate', 'Unknown')
    species_manual_id = metadata.get('Species Manual ID', 'Unknown')
    timestamp = metadata.get('Timestamp', 'Unknown')
    host_device = metadata.get('BATREC|Host Device', 'Unknown')
    host_os = metadata.get('BATREC|Host OS', 'Unknown')
    illuminance = metadata.get('BATREC|Illuminance', 'Unknown')
    batrec_version = metadata.get('BATREC|Version', 'Unknown')
    kaleidoscope_version = metadata.get('WA|Kaleidoscope|Version', 'Unknown')

    # Formatting the timestamp if it's a datetime object
    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    # Printing parsed metadata
    parsed_metadata = {
        'GUANO Version': version,
        'Length (seconds)': length,
        'Latitude': latitude,
        'Longitude': longitude,
        'Location (Lat, Lon)': location,
        'Elevation (meters)': elevation,
        'Device Make': make,
        'Device Model': model,
        'Original Filename': original_filename,
        'Sample Rate': samplerate,
        'Bit Depth': bit_depth,
        'Species Manual ID': species_manual_id,
        'Timestamp': timestamp,
        'Host Device': host_device,
        'Host OS': host_os,
        'Illuminance (lux)': illuminance,
        'BatRecorder Version': batrec_version,
        'Kaleidoscope Version': kaleidoscope_version

    }

    return parsed_metadata

def extract_guano_metadata(file_path):
    with wave.open(file_path, "rb") as wf:
        sample_width = wf.getsampwidth()  # Sample width in bytes
        bit_depth = sample_width * 8  # Convert to bits

    try:
        # Open the GUANO file
        guano_file = guano.GuanoFile(file_path)

        # Print all the metadata values
        metadata = {}
        for key, value in guano_file.items():
            metadata[key] = value

        metadata["Bit Depth"] = bit_depth

        if metadata:
            return parse_guano_metadata(metadata)
        else:
            return "No metadata found in this GUANO file"
    except Exception as e:
        return f"Error: {e}"

def mean_and_std_time_difference(call_parameters):

    call_number = len(call_parameters)

    # Check if there are enough elements to calculate differences
    if call_number < 2:
        return {
        'Number of call': call_number,
        'TBC mean diff': 0,
        'TBC standard deviation': 0
        }


    # Extract t_start_in_file from each Call_parameter object
    start_times = [param.t_start_in_file for param in call_parameters]

    # Calculate the time differences between consecutive start times
    time_differences = [start_times[i+1] - start_times[i] for i in range(len(start_times) - 1)]

    # Calculate the mean and standard deviation of the time differences
    mean_difference = np.mean(time_differences)
    std_difference = np.std(time_differences)

    General_file_data = {
        'Number of call': call_number,
        'TBC mean diff': mean_difference,
        'TBC standard deviation': std_difference
        }

    return General_file_data

def get_all_wav_data(wav_file_path):
    # Extract metadata from the WAV file
    wav_metadata = extract_guano_metadata(wav_file_path)

    # Process the WAV file
    processing_result = process_wav_file(wav_file_path)  # Assuming this function processes the file and returns a result
    #print('processing_result',processing_result)
    # Extract time difference statistics
    wav_extracted_data = mean_and_std_time_difference(processing_result)  # Assuming final_result contains Call_parameter objects

    # Store all data in a single dictionary
    all_wav_data = {
        'wav_metadata': wav_metadata,
        'wav_extracted_data': wav_extracted_data,
        'processing_result': processing_result
    }

    return all_wav_data

import pandas as pd

def extract_attributes_from_call_param(call_param):
    """
    Extract attributes from a Call_parameter instance and return them as a dictionary.
    """
    # Convert the attributes of the Call_parameter object to a dictionary
    attributes = {attr: getattr(call_param, attr) for attr in dir(call_param) if not callable(getattr(call_param, attr)) and not attr.startswith("__")}
    return attributes

def flatten_dict(d):
    """
    Flatten a nested dictionary into a single-level dictionary.
    """
    if not isinstance(d, dict):  # Ensure d is a dictionary
        return {}

    if not d:  # Check if d is None or empty
        return {}

    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):  # If the value is a dictionary, recursively flatten it
            flat_dict.update(flatten_dict(value))
        else:
            flat_dict[key] = value
    return flat_dict

def export_to_csv(all_wav_data, output_file):
    """
    Export the wav metadata, extracted data, and processing results to a CSV file.
    Each parameter from processing_result gets its own column in the CSV file.

    Parameters:
    - all_wav_data: Dictionary containing 'wav_metadata', 'wav_extracted_data', and 'processing_result'.
    - output_file: The file path where the CSV should be saved.
    """
    # Flatten the wav_metadata and wav_extracted_data
    wav_metadata_flat = flatten_dict(all_wav_data['wav_metadata'])
    wav_extracted_data_flat = flatten_dict(all_wav_data['wav_extracted_data'])

    # Initialize a list to hold all rows of data
    rows = []

    # Get the processing result (assuming it's a list of Call_parameter objects)
    processing_result = all_wav_data['processing_result']

    # Loop through each Call_parameter instance and create a row
    if processing_result:  # Ensures processing_result is not None and not empty
        for result in processing_result:
            # Extract attributes from Call_parameter object
            result_flat = extract_attributes_from_call_param(result)

            # Combine metadata, extracted data, and processing results into one row
            row = {**wav_metadata_flat, **wav_extracted_data_flat, **result_flat}

            # Append the row to the list
            rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(rows)

    # Export to CSV
    df.to_csv(output_file, index=False)



def process_all_wav_files2(directory: str, output_file: str, *, verbose: bool = True, recursive: bool = False):
    """Batch-process WAV files in *directory* and append per-call rows to *output_file* (CSV).

    - One row per detected call.
    - Writes after each WAV so you don't lose progress.

    Requires pandas (pd). If pandas isn't installed, raise a clear error.
    """

    if pd is None:
        raise ImportError("pandas is required for CSV export. Install with: pip install pandas")

    directory = os.path.abspath(directory)
    output_file = os.path.abspath(output_file)

    file_exists = os.path.exists(output_file)

    def iter_wavs():
        if not recursive:
            for name in os.listdir(directory):
                if name.lower().endswith('.wav'):
                    yield os.path.join(directory, name)
        else:
            for root, _, files in os.walk(directory):
                for name in files:
                    if name.lower().endswith('.wav'):
                        yield os.path.join(root, name)

    for file_path in iter_wavs():
        filename = os.path.basename(file_path)
        if verbose:
            print(f"Processing {filename}...")

        try:
            all_wav_data = get_all_wav_data(file_path)
        except Exception as exc:
            # File-level failures (loading, candidate detection, metadata, ...)
            # must not interrupt recursive/batch processing.
            if verbose:
                print(
                    f"Skipping file {filename} after "
                    f"{type(exc).__name__}: {exc}"
                )
            continue

        wav_metadata_flat = flatten_dict(all_wav_data.get('wav_metadata'))
        wav_extracted_data_flat = flatten_dict(all_wav_data.get('wav_extracted_data'))
        processing_result = all_wav_data.get('processing_result') or []

        file_results = []
        for result in processing_result:
            result_flat = extract_attributes_from_call_param(result)
            row = {"filename": filename, **wav_metadata_flat, **wav_extracted_data_flat, **result_flat}
            file_results.append(row)

        if file_results:
            df = pd.DataFrame(file_results)
            df.to_csv(output_file, mode='a', index=False, header=not file_exists)
            file_exists = True


def process_directory(directory_path: str, output_csv_path: str | None = None, *, recursive: bool = False, verbose: bool = True) -> str:
    """Convenience wrapper.

    Parameters
    - directory_path: folder containing WAV files
    - output_csv_path: optional explicit output CSV path
    - recursive: also scan subfolders
    - verbose: print per-file progress

    Returns the output CSV path.
    """
    directory_path = os.path.abspath(directory_path)
    directory_name = os.path.basename(directory_path.rstrip(os.sep))

    if output_csv_path is None:
        output_csv_path = os.path.join(directory_path, f"wav_processing_results_{directory_name}.csv")

    process_all_wav_files2(directory_path, output_csv_path, verbose=verbose, recursive=recursive)
    return output_csv_path
