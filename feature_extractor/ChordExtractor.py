import os
import tensorflow as tf
import numpy as np
from music21 import midi, stream, note, chord
import multiprocessing as mp
from typing import Union
from functools import partial
from tqdm import tqdm
from feature_extractor.utils.constants import CHORDS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # removes unwanted info about tf CPU operations

class MModelFinal(tf.keras.Model):
    
    def __init__(self, track_role="accompaniment", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.__labels_embedding = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(64, activation="tanh")
        )
        self.__features_embedding = tf.keras.layers.Dense(64, activation="tanh")
        self.__post_feat_embedding = tf.keras.layers.Dense(64, activation="tanh")        
        self.__input_norm = tf.keras.layers.LayerNormalization(axis=-1)
                
        self.__GRU1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, return_sequences=True)
        )
        self.__LN1 = tf.keras.layers.LayerNormalization(axis=-1)

        self.__GRU2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128)
        )
        self.__LN2 = tf.keras.layers.LayerNormalization(axis=-1)
        self.__dropout = tf.keras.layers.Dropout(0.3)
        self.__dense2 = tf.keras.layers.Dense(128, activation="relu")
        self.__dense3 = tf.keras.layers.Dense(24, activation = "softmax")
        
        self._name = "m_model_final"
        self.__track_role = track_role
        self.compile()
        path = os.path.join(os.getcwd(), "feature_extractor", "chex_ckpt")
        self.load_weights(
            os.path.join(
                path,
                f"ckpt_{track_role}_"
                )
            ).expect_partial()
        
    @property
    def track_role(self):
        return self.__track_role
    
    @track_role.setter
    def track_role(self, track_role: str):
        """set the kind of track role the model should predict (one of accompaniment, main_melody, bass) 
        and load the respective weights

        Args:
            track_role (str): the track role selected (one of accompaniment, main_melody, bass)
        """
        self.__track_role = track_role
        path = os.path.join(os.getcwd(), "feature_extractor", "chex_ckpt")
        self.load_weights(
            os.path.join(
                path,
                f"ckpt_{track_role}_"
                )
            ).expect_partial()
        
    def preprocess(self, file_paths: list, initial_conditions = [], num_workers=1) -> dict:
        """Preprocess a series of midi files with optionally multiprocessing

        Args:
            file_paths (list): list of path to the relative midis
            initial_conditions (list, optional): series of 3 chords representing the initial conditions. Defaults to [].
            num_workers (int, optional): number of processes for multiprocessing (if num_workers=1 no multiprocessing is used). Defaults to 1.

        Returns:
            dict: dictionary of features to pass to self.predict
        """
        
        assert len(initial_conditions) <= 3, "maximum number of chords for initial condition is 3"
        
        feat_dict = dict()
        reordering_indexes = list()
        initial_conditions = [CHORDS.index(chord_) for chord_ in initial_conditions]
        
        if num_workers == 1:        
            
            for file_path in file_paths:
                
                pre_feat, features = _instance_preprocess(file_path, initial_conditions=initial_conditions)
                
                if features.shape[0] not in feat_dict.keys():
                    feat_dict[features.shape[0]] = [list(), list()]
                    
                feat_dict[features.shape[0]][0].append(
                    pre_feat
                    )
                    
                feat_dict[features.shape[0]][1].append(
                    np.expand_dims(features, axis=0)
                    )
                
                reordering_indexes.append(features.shape[0])
                
        else:
            with mp.Pool(num_workers) as pool:
                for pre_feat, features in tqdm(pool.imap(partial(_instance_preprocess, initial_conditions=initial_conditions), iter(file_paths)), total=len(file_paths)):
                    
                    if features.shape[0] not in feat_dict.keys():
                        feat_dict[features.shape[0]] = [list(), list()]
                        
                    feat_dict[features.shape[0]][0].append(
                        pre_feat
                        )
                        
                    feat_dict[features.shape[0]][1].append(
                        np.expand_dims(features, axis=0)
                        )
                    
                    reordering_indexes.append(features.shape[0])
        
        
        for key in feat_dict.keys():
            feat_dict[key][0] = np.concatenate(feat_dict[key][0], axis=0)
            feat_dict[key][1] = np.concatenate(feat_dict[key][1], axis=0)
        
        
        return {
            "features" : feat_dict,
            "reordering_indexes" : reordering_indexes
        }
    
    def __logits_to_chords(self, encodings: np.array):
        
        """converts logits to chords name

        Returns:
            chords (list): list of chords's names
        """
                
        chords = list()
        chords_in = list()
        argmax_indexes = np.argmax(encodings, axis=-1)
        for indexes in argmax_indexes:
            for index in indexes:
                chords_in.append(CHORDS[index])                
            chords.append(chords_in)
            chords_in = list()
        return chords
    
    def __logits_to_one_hot(self, logits):
        
        """converts logits to one hot encodings of chords

        Returns:
            one_hot (np.array): one hot encoding of the chords
        """
        
        argmax_indexes = np.argmax(logits, axis=-1)
        one_hot = np.eye(24)[argmax_indexes]
        return one_hot
        
            
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        
        pre_lab, features = inputs
        outputs = list()
        in_shape = tf.shape(features)
        features = tf.concat([features, tf.zeros((in_shape[0],1,12))], axis=1)
        pre_lab_emb = self.__labels_embedding(pre_lab)
        features_emb = tf.expand_dims(
            self.__features_embedding(features[:, 0, :]),
            axis=1
        )
        post_feat_emb = tf.expand_dims(
            self.__post_feat_embedding(features[:, 1, :]),
            axis=1
        )
        in_emb = tf.concat([pre_lab_emb, features_emb, post_feat_emb], axis=1)
        in_emb = self.__input_norm(in_emb)
        x = self.__GRU1(in_emb)
        x = self.__LN1(x)
        x = self.__dropout(x)
        
        x = self.__GRU2(x)
        x = self.__LN2(x)
        x = self.__dropout(x)
        
        x = self.__dense2(x)
        x = self.__dropout(x)
        
        out = self.__dense3(x)

        c_index = tf.math.argmax(out, axis=-1)
        one_hot_out = tf.one_hot(c_index, depth=out.shape[-1])
        
        outputs.append(tf.expand_dims(out, axis=1))
        pre_lab = tf.concat([pre_lab, tf.expand_dims(one_hot_out, axis=1)], axis=1)
        
        for index in range(1,features.shape[1] - 1):
            pre_lab_emb = self.__labels_embedding(pre_lab[:,-3:, :])
            features_emb = tf.expand_dims(
                self.__features_embedding(features[:, index, :]),
                axis=1
            )
            post_feat_emb = tf.expand_dims(
                self.__post_feat_embedding(features[:, index+1, :]),
                axis=1
            )
            in_emb = tf.concat([pre_lab_emb, features_emb, post_feat_emb], axis=1)
            in_emb = self.__input_norm(in_emb)
            
            x = self.__GRU1(in_emb)
            x = self.__LN1(x)
            x = self.__dropout(x)
            
            x = self.__GRU2(x)
            x = self.__LN2(x)
            x = self.__dropout(x)
            
            x = self.__dense2(x)
            x = self.__dropout(x)
            
            out = self.__dense3(x)
            
            c_index = tf.math.argmax(out, axis=-1)
            one_hot_out = tf.one_hot(c_index, depth=out.shape[-1])
            outputs.append(tf.expand_dims(out,axis=1))
            pre_lab = tf.concat([pre_lab, tf.expand_dims(one_hot_out, axis=1)], axis=1)
        
        outputs =  tf.concat(outputs, axis=1)
        return outputs 
    
    def predict(self, inputs, return_type="chords"):
        
        """
        Predicts chord from features inputs
        
        Params:
            inputs (np.array): the input features of the model
            return_type (str): one of [chords, one_hot, logits], select the format of the returned predictions

        Returns:
            chords predictions (np.array): base on the return_type selection returns logits, one hot encoding or names of predicted chords
        """
        
        assert return_type in ["chords", "one_hot", "logits"], "return_type should be one of [chords, one_hot, logits]"
        
        pred_dict = dict()
        args = inputs["features"]
        reordering_indexes = inputs["reordering_indexes"]
        predictions = list()
        for key in args.keys():
            total_pred = list()
            
            for i in range(args[key][1].shape[1]):
                logits = self((
                    args[key][0],
                    args[key][1][:,i,:,:]
                ))
                oh_pred = self.__logits_to_one_hot(logits)
                args[key][0] = oh_pred[:,-3:,:]
                total_pred.append(logits)
                
            total_pred = np.concatenate(total_pred, axis=1)
        
            if return_type == "chords":
                pred_dict[key] = self.__logits_to_chords(total_pred)                    
            if return_type == "one_hot":
                pred_dict[key] = self.__logits_to_one_hot(total_pred)
            if return_type == "logits":
                pred_dict[key] = total_pred

        for index in reordering_indexes:
                predictions.append(
                    pred_dict[index].pop(0)
                )
                
        return predictions



#___________PRIVATE_METHODS_FOR_MULTIPROCESSING___________________

def _open_midi(midi_path: str, remove_drums=True) -> stream.Score:
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

def _instance_preprocess(file: Union[str, stream.Score], initial_conditions = []):
        
    """Preprocess the midi file and create the relative entries for the model

    Returns:
        features (np.array): the model's input features
    """
    
    features = list()
    
    assert type(file) in [str, stream.Score], f"file should be a path string or a music21.stream.Score object"
    
    if type(file) == str:    
        midi_ = _open_midi(file)
    elif type(file) == stream.Score:
        midi_ = file
        
    ts_num = midi_.getTimeSignatures()[0].numerator
    ts_den = midi_.getTimeSignatures()[0].denominator
    time_norm_fact = 4 / ts_den
    total_beats = int(ts_num * time_norm_fact)
    
    for measure in midi_.recurse().getElementsByClass(stream.Measure):
        measure_features = np.zeros((12,))
        for element in measure.recurse():
            
            if type(element) == chord.Chord:
                for note_ in element.notes:
                    quarterLength = float(element.quarterLength)
                    pitch_index = note_.pitch.midi % 12
                    measure_features[pitch_index] += (quarterLength/total_beats)*(note_.volume.velocity/127)
                    
            if type(element) == note.Note:
                quarterLength = float(element.quarterLength)
                pitch_index = element.pitch.midi % 12
                measure_features[pitch_index] += (quarterLength/total_beats)*(element.volume.velocity/127)
                
        features.append(measure_features)
    
    adding_steps = (4 - len(features)%4)%4
    for _ in range(adding_steps):
        features.append(np.zeros(12))  
    features = np.array(features)

    new_shape = (features.shape[0]//4, 4, 12)
    features = features.reshape((new_shape))
    
    if len(initial_conditions) == 0:
        initial_conditions = np.zeros((1, 3, 24))
    else:
        initial_conditions = np.expand_dims(np.eye(24)[initial_conditions], axis=0)
        padding = np.zeros((1, 3 - initial_conditions.shape[1], 24))
        initial_conditions = np.concatenate([padding, initial_conditions], axis=1)
    
    return initial_conditions, features

CHORD_EXTRACTOR = MModelFinal()