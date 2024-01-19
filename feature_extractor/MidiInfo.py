import os
from music21 import stream, midi, key, tempo, instrument, chord, note, interval, pitch
from typing import Union
import datetime
import shutil
from pprint import pprint


class MidiInfo():
    
    def __init__(self, midi: Union[str, stream.Score], file_name:str = "", genre:str = None, transpose:bool = True, kwargs:dict=None) -> None:
        
        if type(midi) is str:
            self.__midi_file__ = self.__open_midi(midi)
            if file_name == "":
                self.__file_name__, _ = os.path.splitext(os.path.split(midi)[1])
            else:
                self.file_name = file_name
        elif type(midi) is stream.Score:
            self.__midi_file__ = midi
            if file_name == "":
                self.__file_name__, _ = "midi_" + datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
            else:
                self.__file_name__ = file_name
            
        if kwargs is None:
            self.__bpm = self.get_bpm()[0]
            self.__time_signature = self.get_time_signature()[0]
            self.__min_vel, self.__max_vel = self.get_min_max_velocity()
            self.__num_measures = self.number_of_measures()
            self.__rhythm = self.get_sample_rhythm()
            self.__key = self.get_audio_key()
            self.__instrument = self.get_instrument()
        else:
            self.__bpm = kwargs["bpm"]
            self.__time_signature = kwargs["time_signature"]
            self.__min_vel = kwargs["min_vel"]
            self.__max_vel = kwargs["max_vel"]
            self.__key = kwargs["key"]
            self.__num_measures = kwargs["num_measures"]
            self.__rhythm = kwargs["rhythm"]
            self.__instrument = kwargs["instrument"]
        
        self.__genre = genre if genre is not None else -1
        
        if transpose:
            self.transpose_to_CAm()
        
    @property
    def bpm(self):
        return self.__bpm
    
    @property
    def time_signature(self):
        return self.__time_signature
    
    @property
    def min_vel(self):
        return self.__min_vel
    
    @property
    def max_vel(self):
        return self.__max_vel
    
    @property
    def key(self):
        return str(self.__key).lower().replace(" ", "")
    
    @property
    def num_measures(self):
        return self.__num_measures
    
    @property
    def rhythm(self):
        return self.__rhythm
    
    @property
    def genre(self):
        return self.__genre
    
    @genre.setter
    def genre(self, genre):
        self.__genre = genre
        
    @property
    def instrument(self):
        return self.__instrument
        
    @property
    def midi_file(self) -> str:
        return self.__midi_file__
    
    @midi_file.setter
    def midi_file(self, midi_path: str) -> None:
        self.__midi_file__ = self.__open_midi(midi_path)
        
    @property
    def file_name(self) -> str:
        return self.__file_name__
    
    @file_name.setter
    def file_name(self, file_name: str) -> None:
        self.__file_name__ = file_name
        
    def __open_midi(self, midi_path, remove_drums=True) -> stream.Score:
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

            ms = midi.translate.midiFileToStream(mf) 
            return ms
        
    def __remove_duplicates(self, lst: list) -> list:
        """remove consecutive duplicate elements

        Args:
            lst (list): the list with duplicates

        Returns:
            list: the list without consecutive duplicates
        """
        temp = list()
        for i in range(len(lst)):
            if lst[i] not in temp:
                temp.append(lst[i])
            else:
                if i != 0 and lst[i-1] != lst[i]:
                    temp.append(lst[i])
                    
        return temp
    
    def get_time_signature(self, allow_multiple: bool = True) -> Union[list, str]:
        """
        Returns the time signature (or a list of time signature) of the passed MIDI file

        Params
        --------

        - midi_file (music21.stream.Score):
        The MIDI file to analyze

        - allow_multiple (bool), optional:   
        Set whether or not consider more than one time signature (Default=True)

        Returns
        -------

        - Time Signatures (list | str): a list containing the time signatures or a single string
        """
        time_sig = list()

        for ts in self.__midi_file__.getTimeSignatures():
            num = ts.numerator
            den = ts.denominator
            time_sig.append(f"{num}/{den}")

            if not allow_multiple:
                return time_sig[0]
        
        
        return self.__remove_duplicates(time_sig)
    
    def get_bpm(self, allow_multiple: bool = True) -> Union[list, int]:
        """
        Returns the bpm (beats per minute) (or a list of bpms) of the passed MIDI file

        Params
        --------

        - midi_file (music21.stream.Score):
        The MIDI file to analyze

        - allow_multiple (bool), optional:   
        Set whether or not consider more than one bpm (Default=True)

        Returns
        -------

        - bpms (list | int): a list containing the bpm or a single integer
        """
        bpms = []

        for bpm in self.__midi_file__.recurse().getElementsByClass(tempo.MetronomeMark):
            bpms.append(int(bpm.number))

            if not allow_multiple:
                return bpms[0]
            
        return self.__remove_duplicates(bpms)
    
    def number_of_measures(self) -> int:
        """Returns the number of measure inside the MIDI file

        Returns:
            int: The number of measures
        """
        measures = list()
        for p in self.__midi_file__.parts.stream():
            measures.append(len(p.recurse().getElementsByClass(stream.Measure)))
            
        return max(measures)
    
    def get_sample_rhythm(self):
        
        """return the type of rhythm present in the midi (standard or triplet)

        Returns:
            sample_rhythm (str): the rhythm of the song
        """
        
        regular_beats = 0
        triplet_beats = 0
        for element in self.__midi_file__.recurse():            
            if type(element) == chord.Chord or type(element) == note.Note:
                if type(element.quarterLength) is float:
                    regular_beats += 1
                    
                else:
                    triplet_beats += 1
                    
        if regular_beats > triplet_beats:
            return "standard"
        else:
            return "triplet"

    def get_min_max_velocity(self):
        """Return the minimum and maximum value of the velocities"""
        min_vel = float("inf")
        max_vel = float("-inf")

        for note_ in self.__midi_file__.semiFlat.notes:
            vel = note_.volume.velocity
            if vel < min_vel:
                min_vel = vel
            if vel > max_vel:
                max_vel = vel

        return (min_vel, max_vel)

    def get_audio_key(self) -> str:
        "return the audio key (if not present tries to analyze the midi to extract it)"
        if self.__midi_file__.keySignature is not None:
            return self.__midi_file__.keySignature
        else:
            key = self.__midi_file__.analyze('key')
            return key

    def list_instruments(self) -> list:
        """returns a list of the instrument present"""
        instruments = list()
        for p in self.__midi_file__.parts.stream():
            if p is not None:
                instruments.append(p.partName)
        return instruments
    
    def __get_tracks__(self) -> list:
        """Get individual tracks in midi stram

        Returns:
            list: the list of parts
        """
        parts = list()
        for p in self.__midi_file__.parts.stream():
            if p is not None:
                parts.append(p)
        return parts
    
    def __prepare_midi__(self, sample_measures, measures)-> midi.MidiFile:
        """create a midi file from a sequence of measures of a parent midi stream

        Args:
            sample_measures: dictionary containing informations for formatting
            measures: wanted measures

        Returns:
            midi.MidiFile: the extracted sample
        """
        midi_stream = stream.Score()
        initial_measure_number = sample_measures[0].number
        initial_measure_offset = sample_measures[0].offset

        key_flag = False if len(sample_measures[0].getElementsByClass(key.KeySignature)) != 0 else True
        tempo_flag = False if len(sample_measures[0].getElementsByClass(tempo.MetronomeMark)) != 0 else True
        inst_flag = False if len(sample_measures[0].getElementsByClass(instrument.Instrument)) != 0 else True

        for measure_index in reversed(range(sample_measures[0].number)):
            reference_measure = measures[measure_index]
            tempo_ = list(reference_measure.getElementsByClass(tempo.MetronomeMark))
            if tempo_flag and len(tempo_) != 0:
                sample_measures[0].insert(0,tempo_[0])
                tempo_flag = False
            inst_ = list(reference_measure.getElementsByClass(instrument.Instrument))
            if inst_flag and  len(inst_) != 0:
                sample_measures[0].insert(0,inst_[0])
                inst_flag = False
            # key_ = list(reference_measure.getElementsByClass(key.KeySignature))
            # if key_flag and len(key_) != 0:
            #     sample_measures[0].insert(0,key_[0])
            #     key_flag = False

            if not inst_flag and not tempo_flag:
                break
            
        for measure in sample_measures:
            measure.number -= initial_measure_number
            measure.offset -= initial_measure_offset

        midi_stream.append(stream.Part(sample_measures))
        return midi_stream 
    
    def save(self, storing_dir: str = "") -> None:
        midi_file = midi.translate.streamToMidiFile(self.__midi_file__)
        path = os.path.join(
            storing_dir,
            self.__file_name__ + ".mid"
        )                 
        midi_file.open(path, "wb")
        midi_file.write()
        midi_file.close()
        
    def get_instrument(self):
        
        instrument_ = -1
        for element in self.__midi_file__.recurse():
            if "Instrument" in element.classes:
                if instrument_ == -1 or instrument_ == None:
                    instrument_ = element.midiProgram
                    
        return instrument_ if instrument_ != None else -1                
        
    
    def split_and_sample(self, save_midis: bool = True, storing_dir="", to_dict=False) -> list:
        """Search inside the midi stram for repeating sequences and extract the as sample

        Args:
            save_midis (bool, optional): whether or not to save midis, file will be saved at storing_dir. Defaults to True.
            storing_dir (str, optinal): the path where to store the sampled midis

        Returns:
            list: a list of dictionaries containing informations about the sample (len, name, stream)
        """
        tracks_info = list()
        for i, part in enumerate(self.__get_tracks__()):            
            measures = list(part.recurse().getElementsByClass(stream.Measure))
            samples = self.__find_repeating_sequences__(
                [list(measure.notes) for measure in measures]
            )
            
            for j, element in enumerate(samples):                
                sample_measures = measures[element["measure_index"]-1:element["measure_index"] + element["len"]-1]
                midi_stream = self.__prepare_midi__(sample_measures, measures) 
                sample = MidiInfo(
                    midi_stream, 
                    file_name=self.__file_name__ + "_" + str(i)+ "_sample" + str(j),
                )
                if save_midis:
                    sample.save(storing_dir)
                if to_dict:
                    tracks_info.append(sample.to_dict())
                else:
                    tracks_info.append(sample)
                    
        if len(tracks_info) == 0:
            if to_dict:
                tracks_info.append(self.to_dict())
            else:
                tracks_info.append(self)
            if save_midis:
                self.save(storing_dir)
            
        return tracks_info
    
    def __find_repeating_sequences__(self, notes):
        """find repeating sequences of fixed length inside a list of notes

        Args:
            notes (list): the list of notes

        Returns:
            list: a list of dictionaries containing the sub sequences and infos
        """
        repeating_sequences = list()
        repeating_check = list()
        for i in  range(4, len(notes)//2 + 1):
            j=0
            while j <= len(notes) - i:

                if notes[j:j+i] == notes[j+i:j+(i*2)] and len(notes[j:j+i][0]) != 0:

                    if notes[j:j+i] not in repeating_check:
                        repeating_sequences.append({
                            "notes" : notes[j:j+i],
                            "measure_index" : j+1,
                            "len" : i
                        })
                        repeating_check.append(notes[j:j+i])

                    j += i
                
                else:
                    j+=1

        for element in repeating_sequences.copy():
            if element["len"] >= 4:
                if len(self.__find_repeating_sequences__(element["notes"])) != 0:
                    repeating_sequences.remove(element)
                    
        return repeating_sequences
    
    def transpose_to_CAm(self) -> None:
        audio_key_name = str(self.__key).lower().replace(" ", "")
        if audio_key_name != "cmajor" and audio_key_name != "aminor":
            mode = audio_key_name[-5:]
            new_tonic_pitch = "C" if mode == "major" else "A"
            transpose_interval = interval.Interval(self.__key.tonic, pitch.Pitch(new_tonic_pitch))
            self.__midi_file__ = self.__midi_file__.transpose(transpose_interval)
            self.__key = self.get_audio_key()
            self.__midi_file__.keySignature = self.__key
            
    def to_dict(self) -> dict:
        return {
            "bpm" : self.__bpm,
            "time_signature" : self.__time_signature,
            "instrument" : self.__instrument,
            "genre" : self.__genre,
            "key" : self.key,
            "min_vel" : self.__min_vel,
            "max_vel" : self.__max_vel,
            "rhythm" : self.__rhythm,
            "num_measures" : self.__num_measures,
            "file_name" : self.__file_name__ + ".mid"
        }
        