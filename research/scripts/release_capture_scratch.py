"""Release one registered candidate scratch only after scoring and backup audits."""
import argparse
import json
from pathlib import Path
import subprocess

import psutil
from transfer_alignment.candidate_cleanup import release, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    root = args.run.resolve()
    if (root / 'scoring.exit').read_text().strip() != '0':
        raise ValueError('Scoring did not exit successfully')
    backup = json.loads((root / 'backup-audit.json').read_text())
    if backup.get('status') != 'passed':
        raise ValueError('Local input backup audit is missing')
    # The local backup must match the current persistent capture and receipts.
    required = {'inputs/fixed-update-input.pt', 'inputs/capture.json',
                'export-receipt/byte-exact/delta-manifest.json'}
    if not required.issubset(backup['files']):
        raise ValueError('Incomplete local backup identities')
    for name in required:
        if sha(root / name) != backup['files'][name]:
            raise ValueError('Persistent input differs from verified local backup')
    owner = json.loads((root / 'launcher.json').read_text())['owner']

    def no_live_workers():
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            return False
        for process in psutil.process_iter():
            try:
                if process.environ().get('PRAXIS_BASELINE_OWNER') == owner:
                    return False
                command = process.cmdline()
                if 'transfer_alignment.compact_choice_measure' in command:
                    return False
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                # Liveness could not be established; do not authorize deletion.
                return False
        return True

    print(json.dumps(release(root, temporary_base='/tmp', no_live_workers=no_live_workers), indent=2))


if __name__ == '__main__':
    main()
