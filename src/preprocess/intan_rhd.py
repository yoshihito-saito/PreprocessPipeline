from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


INTAN_RHD_MAGIC = 0xC6912702


@dataclass
class IntanRhdHeader:
    source_path: Path
    sample_rate: float
    amplifier_sample_rate: float
    aux_input_sample_rate: float
    supply_voltage_sample_rate: float
    board_adc_sample_rate: float
    board_dig_in_sample_rate: float
    num_temp_sensor_channels: int
    num_amplifier_channels: int
    num_aux_input_channels: int
    num_supply_voltage_channels: int
    num_board_adc_channels: int
    num_board_dig_in_channels: int
    num_board_dig_out_channels: int


def _read_qstring(fid) -> str:
    (length_bytes,) = struct.unpack("<I", fid.read(4))
    if length_bytes == 0xFFFFFFFF:
        return ""
    if length_bytes == 0:
        return ""

    payload = fid.read(length_bytes)
    if len(payload) != length_bytes:
        raise ValueError("Unexpected EOF while reading QString from info.rhd")
    try:
        return payload.decode("utf-16-le")
    except UnicodeDecodeError:
        # Keep parser robust for non-standard headers.
        return payload.decode("utf-16-le", errors="ignore")


def read_intan_rhd_header(rhd_path: Path) -> IntanRhdHeader:
    rhd_path = Path(rhd_path)
    with open(rhd_path, "rb") as fid:
        (magic_number,) = struct.unpack("<I", fid.read(4))
        if magic_number != INTAN_RHD_MAGIC:
            raise ValueError(f"Unrecognized Intan file magic for {rhd_path}")

        (ver_major, ver_minor) = struct.unpack("<hh", fid.read(4))
        num_samples_per_data_block = 60 if ver_major == 1 else 128

        (sample_rate,) = struct.unpack("<f", fid.read(4))
        # dsp_enabled + actual/desired frequencies/bandwidths.
        fid.read(26)
        # notch mode + desired/actual impedance test frequency
        fid.read(2 + 8)

        # notes
        _ = _read_qstring(fid)
        _ = _read_qstring(fid)
        _ = _read_qstring(fid)

        num_temp_sensor_channels = 0
        if (ver_major == 1 and ver_minor >= 1) or (ver_major > 1):
            (num_temp_sensor_channels,) = struct.unpack("<h", fid.read(2))

        if (ver_major == 1 and ver_minor >= 3) or (ver_major > 1):
            # eval_board_mode
            fid.read(2)

        if ver_major > 1:
            _ = _read_qstring(fid)  # reference_channel

        (number_of_signal_groups,) = struct.unpack("<h", fid.read(2))

        n_amp = 0
        n_aux = 0
        n_supply = 0
        n_adc = 0
        n_dig_in = 0
        n_dig_out = 0

        for _group in range(number_of_signal_groups):
            _ = _read_qstring(fid)  # signal_group_name
            _ = _read_qstring(fid)  # signal_group_prefix
            signal_group_enabled, signal_group_num_channels, _ = struct.unpack("<hhh", fid.read(6))

            if signal_group_num_channels <= 0 or signal_group_enabled <= 0:
                continue

            for _ch in range(signal_group_num_channels):
                _ = _read_qstring(fid)  # native_channel_name
                _ = _read_qstring(fid)  # custom_channel_name
                # native_order, custom_order, signal_type, channel_enabled, chip_channel, board_stream
                _, _, signal_type, channel_enabled, _, _ = struct.unpack("<hhhhhh", fid.read(12))
                # trigger settings
                fid.read(8)
                # impedance magnitude / phase
                fid.read(8)

                if not channel_enabled:
                    continue
                if signal_type == 0:
                    n_amp += 1
                elif signal_type == 1:
                    n_aux += 1
                elif signal_type == 2:
                    n_supply += 1
                elif signal_type == 3:
                    n_adc += 1
                elif signal_type == 4:
                    n_dig_in += 1
                elif signal_type == 5:
                    n_dig_out += 1

    return IntanRhdHeader(
        source_path=rhd_path,
        sample_rate=float(sample_rate),
        amplifier_sample_rate=float(sample_rate),
        aux_input_sample_rate=float(sample_rate) / 4.0,
        supply_voltage_sample_rate=float(sample_rate) / float(num_samples_per_data_block),
        board_adc_sample_rate=float(sample_rate),
        board_dig_in_sample_rate=float(sample_rate),
        num_temp_sensor_channels=int(num_temp_sensor_channels),
        num_amplifier_channels=int(n_amp),
        num_aux_input_channels=int(n_aux),
        num_supply_voltage_channels=int(n_supply),
        num_board_adc_channels=int(n_adc),
        num_board_dig_in_channels=int(n_dig_in),
        num_board_dig_out_channels=int(n_dig_out),
    )
