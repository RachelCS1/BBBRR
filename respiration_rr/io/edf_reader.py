"""
EDF (European Data Format) reader for REMbo / TEMEC PSG files.

Faithful port of parseEDFHeader / parseEDFChannel from
"Breath by breath RR - Poly + REMbo.html" (@1235-1313).

EDF layout:
  - 256-byte main header
  - 256 * nSignals bytes of per-signal headers
  - data records: for each record, all signals concatenated, each as
    int16 little-endian samples. Digital -> physical via a per-channel
    linear scale.

No external dependency (pyedflib not required); this matches the JS parser
byte-for-byte so behaviour is identical.
"""

from dataclasses import dataclass
import numpy as np


def _ascii(buf: bytes, off: int, length: int) -> str:
    return buf[off:off + length].decode("ascii", errors="replace").strip()


@dataclass
class EDFMeta:
    num_records: int
    rec_duration: float
    num_signals: int
    labels: list
    phys_min: list
    phys_max: list
    dig_min: list
    dig_max: list
    samples_per_rec: list
    fs_per_channel: list
    data_offset: int


class EDFReader:
    """Reads an EDF file's header and demultiplexes individual channels."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            self._bytes = f.read()
        self.meta = self._parse_header()

    def _parse_header(self) -> EDFMeta:
        buf = self._bytes
        num_records = int(_ascii(buf, 236, 8))
        rec_duration = float(_ascii(buf, 244, 8))
        num_signals = int(_ascii(buf, 252, 4))
        if not (num_signals > 0) or not (rec_duration > 0):
            raise ValueError("Invalid EDF header")

        sig_hdr_bytes = 256 * num_signals
        sig = buf[256:256 + sig_hdr_bytes]

        labels, phys_min, phys_max, dig_min, dig_max, samples_per_rec = [], [], [], [], [], []
        off = 0
        for _ in range(num_signals):
            labels.append(_ascii(sig, off, 16)); off += 16
        off += 80 * num_signals            # transducer type
        off += 8 * num_signals             # physical dimension
        for _ in range(num_signals):
            phys_min.append(float(_ascii(sig, off, 8))); off += 8
        for _ in range(num_signals):
            phys_max.append(float(_ascii(sig, off, 8))); off += 8
        for _ in range(num_signals):
            dig_min.append(float(_ascii(sig, off, 8))); off += 8
        for _ in range(num_signals):
            dig_max.append(float(_ascii(sig, off, 8))); off += 8
        off += 80 * num_signals            # prefiltering
        for _ in range(num_signals):
            samples_per_rec.append(int(_ascii(sig, off, 8))); off += 8
        # reserved (32 * nSignals) left unread

        fs_per_channel = [s / rec_duration for s in samples_per_rec]
        data_offset = 256 + sig_hdr_bytes
        return EDFMeta(num_records, rec_duration, num_signals, labels,
                       phys_min, phys_max, dig_min, dig_max,
                       samples_per_rec, fs_per_channel, data_offset)

    def channel_labels(self):
        return list(self.meta.labels)

    def read_channel(self, label: str):
        """Return (time, signal, fs) for a channel, scaled digital -> physical.

        Vectorised equivalent of parseEDFChannel: reshape the interleaved
        record block and slice out the requested channel's samples.
        """
        m = self.meta
        if label not in m.labels:
            raise ValueError(f'EDF channel "{label}" not found. Available: {m.labels}')
        ch = m.labels.index(label)

        total_per_rec = int(sum(m.samples_per_rec))
        ch_samples = m.samples_per_rec[ch]
        rec_offsets = np.concatenate(([0], np.cumsum(m.samples_per_rec)))
        ch_start = int(rec_offsets[ch])

        n_records = m.num_records
        raw = np.frombuffer(
            self._bytes,
            dtype="<i2",
            count=total_per_rec * n_records,
            offset=m.data_offset,
        ).reshape(n_records, total_per_rec)

        digital = raw[:, ch_start:ch_start + ch_samples].astype(np.float64).ravel()

        scale = (m.phys_max[ch] - m.phys_min[ch]) / (m.dig_max[ch] - m.dig_min[ch])
        offset = m.phys_min[ch] - scale * m.dig_min[ch]
        signal = digital * scale + offset

        fs = m.fs_per_channel[ch]
        time = np.arange(signal.size) / fs
        return time, signal, fs

    def pick_activity_channel(self, keywords=("activ", "actigraph")):
        """Auto-pick a REMbo Activity-like channel (_pickEdfActivityChannel @668)."""
        for lbl in self.meta.labels:
            low = lbl.lower()
            if any(k in low for k in keywords):
                return lbl
        return None


def read_edf_channel(path: str, label: str):
    return EDFReader(path).read_channel(label)
