"""Classify recording paths before deciding whether SSD read-ahead helps."""

import ctypes
import os
from pathlib import Path


_NETWORK_FILESYSTEMS = {
    "nfs", "nfs4", "cifs", "smbfs", "smb3", "sshfs", "fuse.sshfs",
    "fuse.rclone", "fuse.s3fs", "davfs", "fuse.davfs", "lustre",
    "beegfs", "ceph", "glusterfs", "9p",
}


def _linux_mount_type(path: Path) -> str | None:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return None
    best_length, best_type = -1, None
    try:
        with mountinfo.open(encoding="utf-8") as stream:
            for line in stream:
                before, separator, after = line.partition(" - ")
                if not separator:
                    continue
                fields = before.split()
                filesystem = after.split()
                if len(fields) < 5 or not filesystem:
                    continue
                mount = fields[4].replace("\\040", " ").replace("\\011", "\t")
                if (str(path) == mount or str(path).startswith(mount.rstrip("/") + "/")) and len(mount) > best_length:
                    best_length, best_type = len(mount), filesystem[0].lower()
    except OSError:
        return None
    return best_type


def _windows_drive_type(drive: str) -> int:
    get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint
    return get_drive_type(drive)


def is_network_source(filename: str | os.PathLike) -> bool:
    """Return true for shares/mounts; unknown Windows drives stay cacheable."""
    raw = os.fspath(filename)
    if os.name == "nt":
        if raw.startswith(("\\\\", "//")):
            return True
        drive = Path(raw).absolute().anchor
        if not drive:
            return True
        # DRIVE_REMOTE=4; unknown/unavailable drives retain the safe cache path.
        return _windows_drive_type(drive) not in (2, 3, 5, 6)
    path = Path(raw).absolute()
    filesystem = _linux_mount_type(path)
    if filesystem is None:
        try:
            import psutil
            mounts = psutil.disk_partitions(all=True)
            match = max((part for part in mounts if
                         str(path) == part.mountpoint or
                         str(path).startswith(part.mountpoint.rstrip("/") + "/")),
                        key=lambda part: len(part.mountpoint), default=None)
            filesystem = match.fstype.lower() if match else None
        except (ImportError, OSError):
            pass
    return filesystem in _NETWORK_FILESYSTEMS
