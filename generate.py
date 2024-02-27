import argparse
from datetime import datetime
from pathlib import Path

import yaml

from commu_dset import DSET
#from commu_wrapper import make_midis
from musicomb import MusiComb

def main(args: argparse.Namespace, timestamp: str) -> None:

    if args.generate_samples:
        pass
        # role_to_midis = make_midis(
        #     args.bpm,
        #     args.key,
        #     args.time_signature,
        #     args.num_measures,
        #     args.genre,
        #     args.rhythm,
        #     args.chord_progression,
        #     timestamp)
    else:
        role_to_midis = DSET.sample_midis(
            args.bpm,
            args.key,
            args.time_signature,
            args.num_measures,
            args.genre,
            args.rhythm,
            args.chord_progression)
    
    MusiComb(role_to_midis, timestamp, args.bpm, args.time_signature, args.num_measures, args.genre, args.music_length).solve()


if __name__ == '__main__':
    with open('cfg/metadata_nv.yaml') as f:
        meta = yaml.safe_load(f)
        
    # now = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    # Path(f'out/{now}').mkdir(parents=True)
    # # with open(f'out/{now}/metadata.yaml', 'w') as f:
    # #     yaml.dump(vars(args), f)

    # main({
    #     "key" : "aminor",
    #     "bpm" : 90,
    #     "time_signature" : "4/4",
    #     "num_measures" : 8,
    #     "genre" : "cinematic",
    #     "rhythm" : "standard",
    #     "chord_progression" : "Am-G-F-C-Dm-Am-A#-Am",
    #     "generate_samples" : False,
    #     "music_length" : 2
    #     }, now)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bpm', 
        dest='bpm', 
        type=int, 
        required=True, 
        choices=meta['bpm'])
    parser.add_argument(
        '--key', 
        dest='key', 
        type=str, 
        required=True, 
        choices=meta['key'])
    parser.add_argument(
        '--time_signature', 
        dest='time_signature', 
        type=str, 
        required=True, 
        choices=meta['time_signature'])
    parser.add_argument(
        '--num_measures', 
        dest='num_measures', 
        type=int, 
        required=True, 
        choices=meta['num_measures'])
    parser.add_argument(
        '--genre', 
        dest='genre', 
        type=str, 
        required=True, 
        choices=meta['genre'])
    parser.add_argument(
        '--rhythm', 
        dest='rhythm', 
        type=str, 
        required=True, 
        choices=meta['rhythm'])
    parser.add_argument(
        '--chord_progression', 
        dest='chord_progression', 
        type=str, 
        required=True, 
        choices=meta['chord_progression'])
    parser.add_argument(
        '--music_length',
        dest='music_length',
        type=int,
        required=True,
        choices=meta['music_length'])
    parser.add_argument(
        '--generate_samples', 
        dest='generate_samples', 
        default=False, 
        action='store_true')
    args = parser.parse_args()

    now = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    Path(f'out/{now}').mkdir(parents=True)
    with open(f'out/{now}/metadata.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    main(args, now)

