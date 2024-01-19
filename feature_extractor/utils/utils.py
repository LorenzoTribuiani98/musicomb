from utils_.constants import MAJOR_CHORDS, MINOR_CHORDS, TRANSPOSE_MAJOR, TRANSPOSE_MINOR

def transpose_chords_progression(chords_progression: str, transpose_interval: int):
    
    new_chords = list()
    
    chords_progression = chords_progression.split("-")
    
    for chord in chords_progression:
        chord = chord.strip()
        if chord[-1] == "m":
            chord_index = (MINOR_CHORDS.index(chord) + transpose_interval)%12
            new_chords.append(MINOR_CHORDS[chord_index])            
        else:
            chord_index = (MAJOR_CHORDS.index(chord) + transpose_interval)%12
            new_chords.append(MAJOR_CHORDS[chord_index])
    
    chords_string = ""
    for chord in new_chords:
        chords_string += chord + "-"
    return chords_string[:-1]


def transpose_chords_to_CA(key: str, chords_progression: str):
    
    mode = key[-5:]
    note = key[:-5]
    if mode == "major":
        transpose_interval = TRANSPOSE_MAJOR[note]
    if mode == "minor":
        transpose_interval = TRANSPOSE_MINOR[note]
        
    return transpose_chords_progression(chords_progression, transpose_interval)


            
        