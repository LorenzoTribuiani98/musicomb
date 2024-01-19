import pickle
from typing import Union
from music21 import stream, midi, chord, note
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class SVMClassifier:

    def __init__(self, midi_paths: list[str] = None) -> None:
        path = os.path.join(os.getcwd(), "feature_extractor", "classifier_ckpt")
        self.__svm__ = pickle.load(open(os.path.join(path, "model_svm.sav"), "rb"))
        if midi_paths is not None:
            self.__midis__ = [self.__open_midi__(midi_path) for midi_path in midi_paths]
        else:
            self.__midis__ = list()
        self.__enc__ = LabelEncoder()
        self.__enc__.classes_ = np.load(os.path.join(path, "classes.npy"), allow_pickle=True)
        self.__scaler__ = pickle.load(open(os.path.join(path, "scaler.pkl"), "rb"))
        self.__measures__ = 0

    def __open_midi__(self, midi_path: str, remove_drums=True) -> stream.Score:
        """Open a midi file and converts it into a music21 score stream

        Args:
            midi_path (str): path to file
            remove_drums (bool, optional): select wheter or not remove drums from track. Defaults to True.

        Returns:
            stream.Score: returns the stream score
        """
        if midi_path is None:
            return None
        else:
            mf = midi.MidiFile()
            mf.open(midi_path)
            mf.read()
            mf.close()
            if (remove_drums):
                for i in range(len(mf.tracks)):
                    mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10] 
            return midi.translate.midiFileToStream(mf)
        
    @property
    def midis(self) -> stream.Score:
        return self.__midis__
    
    @midis.setter
    def midis(self, midis: list[str]) -> None:
        self.__midis__ = [self.__open_midi__(midi_) for midi_ in midis]

    @property
    def measures(self) -> int:
        return self.__measures__
    
    @measures.setter
    def measures(self, measures: int)-> None:
        self.__measures__ = measures
        
        
    def __mean__(self, lst: list, remove_first=True, normalize=True, normalization_base = 12):
        """Mean of a vector

        Args:
            lst (list): the array on wich compute the mean
            remove_first (bool, optional): select wheter or not remove the first element of the array in the computation of the mean. Defaults to True.
            normalize (bool, optional): select wheter or not normalize the outpus according to a specific base described by normalization_base. Defaults to True.
            normalization_base (int, optional): the normalization base. Defaults to 12.

        Returns:
            _type_: _description_
        """
        if (len(lst) == 0) or (len(lst) <2 and remove_first):
            return 0
        else:
            if remove_first:
                lst.pop(0)
                
            mean_ = sum(lst)/len(lst)
            if normalize:
                return mean_%normalization_base
            else:
                return mean_
    
    def __preprocess__(self):
        """preprocess the input midis to create features for predictions

        Returns:
            np.array: list of features
        """
        features = list()
        progress_bar = tqdm(range(len(self.__midis__)))
        for midi in self.__midis__:
            features.append(self.__preprocess__in(midi))
            progress_bar.update(1)
            
        return np.array(features)
    
    def __preprocess__in(self, midi) -> dict:
        min_octave = float("inf")
        max_octave = float("-inf")
        octaves = list()
        chords_durations = list()
        notes_durations = list()
        chord_dist = list()
        dist = list()
        measures = list()
        previous_note_pitch = 0
        n_chords = 0
        n_individual_notes = 0
        notes_per_chord = list()
        
        if type(midi) is stream.Score:
            for p in midi.parts.stream():
                measures.append(len(p.recurse().getElementsByClass(stream.Measure)))
        else:
                measures.append(len(midi.recurse().getElementsByClass(stream.Measure)))
        n_measure = max(measures)
        
        instrument_ = -1
        for element in midi.recurse():
            
            if "Instrument" in element.classes:
                if instrument_ == -1 or instrument_ == None:
                    instrument_ = element.midiProgram
                    
            if type(element) == chord.Chord:
                n_chords += 1
                inner_list = list()
                previous_pitch = 0
                n_notes_chord = 0
                chords_durations.append(float(element.quarterLength))
                for note_ in element.notes:
                    inner_list.append(abs(note_.pitch.ps - previous_pitch))
                    previous_pitch = note_.pitch.ps
                    n_notes_chord += 1
                    
                    octaves.append(note_.octave)
                    if note_.octave < min_octave:
                        min_octave = note_.octave
                        
                    if note_.octave > max_octave:
                        max_octave = note_.octave
                        
                chord_dist.append(self.__mean__(inner_list))
                notes_per_chord.append(n_notes_chord)
                    
            elif type(element) == note.Note:
                n_individual_notes +=1
                dist.append(abs(element.pitch.ps - previous_note_pitch))
                previous_note_pitch = element.pitch.ps
                octaves.append(element.octave)
                if element.octave < min_octave:
                    min_octave = element.octave
                if element.octave > max_octave:
                    max_octave = element.octave
                notes_durations.append(float(element.quarterLength))
                
                
        
        return [
            n_chords / n_measure,
            n_individual_notes / n_measure,
            self.__mean__(chord_dist, remove_first=False, normalize=False),
            self.__mean__(dist),
            self.__mean__(notes_per_chord, remove_first=False, normalize=False),
            min_octave,
            max_octave,
            self.__mean__(octaves, remove_first=False, normalize=False),
            self.__mean__(chords_durations, remove_first=False, normalize=False),
            self.__mean__(notes_durations, remove_first=False, normalize=False),
            -1 if instrument_ == None else instrument_
        ]
    
    
    def predict(self, return_raw:bool=False) -> Union[np.array, str]:
        """generate track role predictions of the selected midi file

        Args:
            return_raw (bool): wheter or not to return raw predictions (logits) or the class string. Default to False.

        Returns:
            str|np.array: the track role classification
        """
        
        features = self.__preprocess__()
        prediction = self.__svm__.predict(
            np.array(
                self.__scaler__.transform(features)
            )
        )
        
        if return_raw:
            return prediction
        else:
            return self.__enc__.inverse_transform(prediction)
        
TRACK_ROLE_CLASSIFIER = SVMClassifier()