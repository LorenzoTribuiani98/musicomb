import argparse
from datetime import datetime
from pathlib import Path

import yaml

from commudset import DSET
from midicomb import MidiComb


def main(args: argparse.Namespace, timestamp: str) -> None:
    role_to_midis = DSET.sample_midis(
        args.bpm,
        args.key,
        args.time_signature,
        args.num_measures,
        args.genre,
        args.rhythm,
        args.chord_progression)

    MidiComb(role_to_midis, timestamp).solve()


if __name__ == "__main__":
    with open('cfg/metadata.yaml') as f:
        meta = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bpm', dest='bpm', 
        type=int, required=True, choices=meta['bpm'])
    parser.add_argument(
        '--key', dest='key', 
        type=str, required=True, choices=meta['key'])
    parser.add_argument(
        '--time_signature', dest='time_signature', 
        type=str, required=True, choices=meta['time_signature'])
    parser.add_argument(
        '--num_measures', dest='num_measures', 
        type=int, required=True, choices=meta['num_measures'])
    parser.add_argument(
        '--genre', dest='genre', 
        type=str, required=True, choices=meta['genre'])
    parser.add_argument(
        '--rhythm', dest='rhythm', 
        type=str, required=True, choices=meta['rhythm'])
    parser.add_argument(
        '--chord_progression', dest='chord_progression', 
        type=str, required=True, choices=meta['chord_progression'])
    args = parser.parse_args()

    now = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    Path(f'out/{now}').mkdir(parents=True)
    with open(f'out/{now}/metadata.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    main(args, now)
