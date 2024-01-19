import numpy as np

CHORDS = [
    "C", 
    "Cm", 
    "C#", 
    "C#m",
    "D", 
    "Dm", 
    "D#", 
    "D#m", 
    "E", 
    "Em",
    "F", 
    "Fm",
    "F#", 
    "F#m", 
    "G", 
    "Gm",
    "G#", 
    "G#m", 
    "A", 
    "Am", 
    "A#", 
    "A#m",
    "B", 
    "Bm"
    ]

MAJOR_INDEXES = [2*i for i in range(12)]
MINOR_INDEXES = [(2*i)+1 for i in range(12)]

MAJOR_CHORDS = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B"
]

MINOR_CHORDS = [
    "Cm",
    "C#m",
    "Dm",
    "D#m",
    "Em",
    "Fm",
    "F#m",
    "Gm",
    "G#m",
    "Am",
    "A#m",
    "Bm"
]

TRANSPOSE_MINOR = {
    "a" : 0,
    "a#" : -1,
    "b" : -2,
    "c" : -3,
    "c#" : -4,
    "d" : -5,
    "d#" : -6,
    "e" : -7,
    "f" : -8,
    "f#" : -9,
    "g" : -10,
    "g#" : -11,
}

TRANSPOSE_MAJOR = {
    "c" : 0,
    "c#" : -1,
    "d" : -2,
    "d#" : -3,
    "e" : -4,
    "f" : -5,
    "f#" : -6,
    "g" : -7,
    "g#" : -8,
    "a" : -9,
    "a#" : -10,
    "b" : -11,
}


CHORDS_ENCODINGS = {}
for i in range(24):
    CHORDS_ENCODINGS[CHORDS[i]] = np.zeros(24)
    CHORDS_ENCODINGS[CHORDS[i]][i] = 1
CHORDS_ENCODINGS[""] = np.zeros(24)
    
NOTES = {
    0 : "C",
    1 : "C#",
    2 : "D",
    3 : "D#",
    4 : "E",
    5 : "F",
    6 : "F#",
    7 : "G",
    8 : "G#",
    9 : "A",
    10 : "A#",
    11 : "B", 
}
    
CHORD_TRIADS = {
    "C" : ["C", "E", "G"],
    "Cm" : ["C", "D#", "G"],
    "C#" : ["C#", "F", "G#"],
    "C#m" : ["C#", "E", "G#"],
    "D" : ["D", "F#", "A"],
    "Dm" : ["D", "F", "A"],
    "D#" : ["D#", "G", "A#"],
    "D#m" : ["D#", "F#", "A#"],
    "E" : ["E", "G#", "B"],
    "Em" : ["E", "G", "B"],
    "F" : ["F", "A", "C"],
    "Fm" : ["F", "G#", "C"],
    "F#" : ["F#", "A#", "C#"],
    "F#m" : ["F#", "A", "C#"],
    "G" : ["G", "B", "D"],
    "Gm" : ["G", "A#", "D"],
    "G#" : ["G#", "C", "D#"],
    "G#m" : ["G#", "B", "D#"],
    "A" : ["A", "C#", "E"],
    "Am" : ["A", "C", "E"],
    "A#" : ["A#", "D", "F"],
    "A#m" : ["A#", "C#", "F"],
    "B" : ["B", "D#", "F#"],
    "Bm" : ["B", "D", "F#"],
}