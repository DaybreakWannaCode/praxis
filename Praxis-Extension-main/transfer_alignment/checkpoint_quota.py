"""Conservative pre-save accounting against an explicit purchased-volume quota.

Reserve the entire replacement while the previous checkpoint remains present.
Not a filesystem reservation: external writers can still race this preflight.
"""
import os
from pathlib import Path
import stat


def accounted_bytes(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('A real volume directory is required')
    seen = set()
    total = 0
    def fail(error): raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        # Count directory allocation too; regular hard links count once.
        for path in [Path(directory)] + [Path(directory) / name for name in files]:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode): continue
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError('Unexpected special file on checkpoint volume')
            key = (info.st_dev, info.st_ino)
            if key in seen: continue
            seen.add(key)
            total += max(info.st_size, info.st_blocks * 512)
    return total


class CheckpointQuota:
    def __init__(self, volume, *, quota_bytes, replacement_bytes, reserve_bytes):
        self.volume = Path(volume)
        if any(type(v) is not int or v <= 0 for v in (quota_bytes, replacement_bytes, reserve_bytes)):
            raise ValueError('Positive explicit quota, replacement bound and reserve required')
        self.quota, self.replacement, self.reserve = quota_bytes, replacement_bytes, reserve_bytes

    def __call__(self, store_root, step):
        volume = self.volume.resolve(strict=True)
        store = Path(store_root).resolve(strict=True)
        if store != volume and volume not in store.parents:
            raise ValueError('Checkpoint store lies outside accounted volume')
        used = accounted_bytes(self.volume)
        needed = used + self.replacement + self.reserve
        if needed > self.quota:
            raise OSError(f'Checkpoint quota guard: {used} used + {self.replacement} replacement '
                          f'+ {self.reserve} reserve exceeds {self.quota} bytes')
        return dict(accounted_bytes=used, replacement_bytes=self.replacement,
                    reserve_bytes=self.reserve, quota_bytes=self.quota, step=step)
