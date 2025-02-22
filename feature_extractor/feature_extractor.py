from midi_utlis import MidiInfo
from ChordExtractor import CHORD_EXTRACTOR
from SVMClassifier import TRACK_ROLE_CLASSIFIER
import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from utils.constants import INITIAL_STATES
import numpy as np

def analyze_midi(inputs, storing_dir=""):
    file_path, genre, instrument = inputs
    midi_file = MidiInfo(file_path)
    midi_file.genre = genre
    midi_file.instrument = instrument
    return midi_file.split_and_sample(storing_dir=storing_dir, to_dict=True)

def analyze_mp(file_paths, storing_dir="", num_workers=8, total_elements=None, filter_output_df=True, minimum_spcp=5):
    metadata = dict()
    
    if type(file_paths) is list:
        total_elements = len(file_paths)
    else:
        assert total_elements is not None, "total_elements is mandatory when using generators"
        
    with mp.Pool(num_workers) as pool:
        for features in tqdm(pool.imap(partial(analyze_midi, storing_dir=storing_dir), file_paths), total=total_elements): #features = analyze_midi(path, storing_dir)   
            for feat in features:    
                for key, value in feat.items():
                    if key not in metadata.keys():
                        metadata[key] = list()                
                    metadata[key].append(value)

    paths = [os.path.join(storing_dir, file_name) for file_name in metadata["file_name"]]
    
    TRACK_ROLE_CLASSIFIER.midis = paths
    print("preprocessing features for Track Role classification")
    track_role_predictions = TRACK_ROLE_CLASSIFIER.predict()
    metadata["track_role"] = track_role_predictions
    
    df =  pd.DataFrame(metadata)
    df_acc = df[(df["track_role"] == "accompaniment") | (df["track_role"] == "riff") | (df["track_role"] == "pad")]
    df_mm = df[(df["track_role"] == "main_melody") | (df["track_role"] == "sub_melody")]
    df_bass = df[df["track_role"] == "bass"]
    
    df_roles = {
        "accompaniment" : df_acc,
        "main_melody" : df_mm,
        "bass" : df_bass
    }
    
    for role in ["accompaniment", "main_melody", "bass"]:
        print(f"Preprocessing {role} features for chord extraction")
        paths = [os.path.join(storing_dir, file_name) for file_name in df_roles[role]["file_name"]]
        CHORD_EXTRACTOR.track_role = role
        if role == "main_melody":
            chords = CHORD_EXTRACTOR.process_n_predict(
                paths,
                initial_conditions = INITIAL_STATES,
                num_workers = num_workers
            )
            df_roles[role] = pd.concat([df_roles[role]]*10, ignore_index=True)
            df_roles[role]["chord_progression"] = np.array(chords).flatten()
        else:
            chords = CHORD_EXTRACTOR.process_n_predict(
                paths,
                num_workers = num_workers
            )
            df_roles[role]["chord_progression"] = chords[0]
        
    metadata = pd.concat([
        df_roles["accompaniment"],
        df_roles["main_melody"],
        df_roles["bass"]],
        ignore_index=True)
    
    metadata = metadata.sort_values("file_name")
    metadata["chord_progression"] = metadata["chord_progression"].apply(lambda x: str(x))
    metadata = metadata.drop_duplicates()
    metadata = metadata.reset_index(drop=True)
    
    if filter_output_df:
        unique_cp = metadata["chord_progression"].unique()
        total_unique = 0
        unique_filtered = 0
        print("Filtering output dataset:")
        for cp in tqdm(unique_cp):
            condition = metadata["chord_progression"] == cp
            if len(metadata[condition]) <= minimum_spcp:
                metadata = metadata[~condition]
            else:
                total_unique += len(metadata[condition])
                unique_filtered += 1
        print()
        print(f"Dataset size:        {len(metadata)}")
        print(f"mean samples per cp: {total_unique/unique_filtered}")
        
    return metadata
        
# if __name__ == "__main__":
    
#     def generator(metadata):
#         for i in range(len(metadata)):
#             yield (
#                 f"dataset/commu_midi/{metadata['split_data'].iloc[i]}/raw/{metadata['id'].iloc[i]}.mid",
#                 metadata["genre"].iloc[i],
#                 metadata["inst"].iloc[i]
#             )
    
#     meta = pd.read_csv("dataset/commu_meta.csv")[:100]
#     df = analyze_mp(generator(meta), total_elements=len(meta), storing_dir="dataset/new_midis", num_workers=8)
#     df.to_csv("dataset/commu_meta_nv.csv")
