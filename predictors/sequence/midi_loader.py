#!/usr/bin/python3
"""A module for the MIDI dataset loader.
"""
import logging
import numpy as np
import string
import collections
import math
from operator import itemgetter

import pretty_midi

from helpers import Lin2WaveEncoder

_SUSTAIN_ON = 0
_SUSTAIN_OFF = 1
_NOTE_ON = 2
_NOTE_OFF = 3

NOTE_ON = 1
NOTE_OFF = 2
TIME_SHIFT = 3
VELOCITY = 4

OCTAVES = 11  # ceil(128 / 12)
KEYS_PER_OCTAVE = 12

MAX_SHIFT_STEPS = 100

MIN_MIDI_VELOCITY = 1
MAX_MIDI_VELOCITY = 127

MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127
PROGRAMS_PER_FAMILY = 8
NUM_PROGRAM_FAMILIES = int((MAX_MIDI_PROGRAM - MIN_MIDI_PROGRAM + 1) / PROGRAMS_PER_FAMILY)

logger = logging.getLogger("few-shot")


class MIDILoader(object):
    """Objects of this class parse MIDI files into a sequence of note IDs
    """
    def __init__(self, max_len, dtype=np.int32, persist=True):
        self.max_len = max_len
        self.dtype = dtype
        self.persist = persist
        self.velocity_encoder = Lin2WaveEncoder(MIN_MIDI_VELOCITY, MAX_MIDI_VELOCITY)
        self.time_encoder = None  # TODO make this better, more robust

    def read(self, filepath):
        """Reads a MIDI file.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/freemidi_data/Tool/lateralus.mid"

        """
        return pretty_midi.PrettyMIDI(filepath)

    def is_song(self, filepath):
        return filepath.endswith('.mid')

    def tokenize(self, midi):
        """Turns a MIDI file into a list of event IDs.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/freemidi_data/Tool/lateralus.mid"
        """
        end_time = np.ceil(midi.get_end_time() * 1000)  # end time in millis

        midi_notes = get_notes(midi, use_drums=False)
        midi_control_changes = get_control_changes(midi, use_drums=False)
        midi_notes = apply_sustain_control_changes(midi_notes, midi_control_changes)
        midi_events = separate_events(midi_notes)
        midi_events = merge_overlapped_pitches(midi_events)
        tokens = encode(midi_events, end_time)
        return tokens

    def detokenize(self, tokens):
        """
        From a list of tokens sorted by increasing time and creates a midi object
        """
        # first decode input
        decoded_tokens = decode(tokens)
        midi = pretty_midi.PrettyMIDI()
        #merge notes on and off into a single event ...
        active_notes = [[None for _ in range(128)] for _ in range(16)]
        instr_notes = {}

        for t in decoded_tokens:
            (program, pitch, action, velocity, time) = t
            if active_notes[program][pitch] is None and action == NOTE_ON:
                active_notes[program][pitch] = t
            elif active_notes[program][pitch] is None and action == NOTE_OFF:
                logger.error("there is a note off without a previous note on")
                print("Error")
            else:  #there is a current note
                (_, _, _, vel, start_time) = active_notes[program][pitch]
                # using velocity from note_on -> don't trust the note_off one
                note = pretty_midi.Note(start_time, time, pitch, vel)
                if program not in instr_notes:
                    instr_notes[program] = pretty_midi.Instrument(program=program)
                instr.notes.append(note)
                #reset position
                active_notes[program][pitch] = None
        # now put all in the midi file
        for instr in instr_notes.values()
            midi.instruments.append(inst)
        return midi

    def encode(self, midi_events, end_time):
        """
        """
        self.time_encoder = Lin2WaveEncoder(0, end_time)


        tokens = []
        for event in midi_events:
            [program, _, pitch, action, vel, event_time] = event
            program_family = int2one_hot( (program - MIN_MIDI_PROGRAM) // PROGRAMS_PER_FAMILY , NUM_PROGRAM_FAMILIES)
            octave = int2one_hot(get_octave(pitch), OCTAVES)
            key_id = int2one_hot(get_key_id(pitch), KEYS_PER_OCTAVE)
            e_action = int2one_hot(action, 2)  # on|off (2 actions)
            e_vel = self.velocity_encoder.encode(vel)  # vector of fourier inspired encoding
            e_time = self.time_encoder.encode(event_time)  # vector of fourier inspired encoding
            tok = (program_family, octave, key_id, e_action, e_vel, e_time)
            tokens.append(tok)
        return tokens

    def decode(self, tokens):
        """
        Decodes the input list of time sorted tokens into a PrettyMidi object for later synthesis
        @param tokens: sorted list of tokens of the shape (program_family, octave, key_id, e_action, e_vel, e_time)
                the time is considered in milliseconds
        @return PrettyMidi object
        """
        #find if we have a time encoder already
        if not self.time_encoder:
            #find last event time -> assume tokens are sorted by time
            end_time = tokens[-1][5]
            self.time_encoder = Lin2WaveEncoder(0, end_time)

        decoded = []
        for t in tokens:
            [prog_family, octave, key_id, e_action, e_vel, e_time] = t
            program = one_hot2int(prog_family) * PROGRAMS_PER_FAMILY + 1  # first instrument of the instrument family
            pitch = one_hot2int(octave_key2midi_id(octave, key_id))
            action = one_hot2int(e_action)
            velocity = self.velocity_encoder.decode(e_vel)
            time = self.time_encoder(e_time)
            decoded.append((program, pitch, action, velocity, time))
        return decoded


def separate_events(midi_notes):
    """
    pretty_midi creates a note event that contains the begining and the end of
    it's execution, here we separate it again in note_on and note_off
    """
    events = []
    for program, instrument, _, note in midi_notes:
        n_on =  (program, instrument, note.pitch, _NOTE_ON,  note.velocity, note.start)
        n_off = (program, instrument, note.pitch, _NOTE_OFF, note.velocity, note.end)
        events.append(n_on)
        events.append(n_off)
    return events


def get_octave(midi_id):
    return int(midi_id / 12)


def get_key_id(midi_id):
    return midi_id%12


def octave_key2midi_id(octave, key_id):
    return octave * 12 + key_id


def int2one_hot(x, vector_dim):
    v = np.zeros(vector_dim)
    v[x] = 1
    return v


def one_hot2int(x):
    """
    Easy and dumb way of decoding a one hot encoder into an integer
    """
    factors = np.zeros_like(x)
    res = 0
    for i in range(1, len(factors)+1 )
        exp = len(factors) - i
        res += 2**exp
    return res


def merge_overlapped_pitches(midi_events):
    """This function resolve note conflicts resulting from merging instruments
    of the same class.
    @param midi_events: list of events composed of:
        (program, instrument, pitch, action,  velocity, time)
    """
    # group_by program code & pitch
    groups = {}  ## program: pitch: [event]
    for e in midi_events:
        prog_k = e[0]
        pitch_k = e[2]
        if not prog_k in groups:
            groups[prog_k] = {}
        grp = groups[prog_k]
        if not pitch_k in grp:
            grp[pitch_k] = []
        grp[pitch_k].append(e)
    # Merge overlaped events
    merged_events = []
    for prog_k, prog_v in groups.items():
        for pitch_k, pitch_v in prog_v.items()
            # Sort by time & note event
            grp = pitch_v.sort(key = operator.itemgetter(5))
            # filter conflicting events
            del_indices = []
            for i in range(1, len(grp)-1):  # assume well formed midi
                # is double note on? or double note off?
                # double note on is forward pass -> erase the second on
                # double note off is backwards pass -> erase first off
                if grp[i-0][3] == grp[i-0][3] or grp[i][3] == grp[i+1][3]:
                    del_indices.append(i)
            #delete duplicates elements
            clean_events = np.delete(grp, del_indices).tolist()
            merged_events.extend(merged_e)
    #sort by event time -> for the training to take place
    merged_events.sort(key=operator.itemgetter(5))
    return merged_events


def apply_sustain_control_changes(midi_notes, midi_control_changes,
                                  sustain_control_number=64):
    """Applies sustain to the MIDI notes by modifying the notes in-place.

    Normally, MIDI note start/end times simply describe e.g. when a piano key
    is pressed. It's possible that the sound from the note continues beyond
    the pressing of the note if a sustain on the instrument is active. The
    activity of sustain on MIDI instruments is determined by certain control
    events. This function alters the start/end time of MIDI notes with respect
    to the sustain control messages to mimic sustain.

    Arguments:
        midi_notes ([(int, int, bool, pretty_midi.Note)]): A list of tuples of
            info on each MIDI note.
        midi_control_changes ([(int, int, bool, pretty_midi.ControlChange)]):
            A list of tuples on each control change event.
    """
    events = []
    events.extend([(midi_note.start, _NOTE_ON, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])
    events.extend([(midi_note.end, _NOTE_OFF, instrument, midi_note) for
      _1, instrument, _2, midi_note in midi_notes])

    for _1, instrument, _2, control_change in midi_control_changes:
        if control_change.number != sustain_control_number:
            continue
        value = control_change.value
        # MIDI spec specifies that >= 64 means ON and < 64 means OFF.
        if value >= 64:
            events.append((control_change.time, _SUSTAIN_ON, instrument,
                           control_change))
        if value < 64:
            events.append((control_change.time, _SUSTAIN_OFF, instrument,
                           control_change))

    events.sort(key=itemgetter(0))

    active_notes = collections.defaultdict(list)
    sus_active = collections.defaultdict(lambda: False)

    time = 0
    for time, event_type, instrument, event in events:
        if event_type == _SUSTAIN_ON:
            sus_active[instrument] = True
        elif event_type == _SUSTAIN_OFF:
            sus_active[instrument] = False
            new_active_notes = []
            for note in active_notes[instrument]:
                if note.end < time:
                    note.end = time
                else:
                    new_active_notes.append(note)
            active_notes[instrument] = new_active_notes
        elif event_type == _NOTE_ON:
            if sus_active[instrument]:
                new_active_notes = []
                for note in active_notes[instrument]:
                    if note.pitch == event.pitch:
                        note.end = time
                        if note.start == note.end:
                            try:
                                midi_notes.remove(note)
                            except ValueError:
                                continue
                    else:
                        new_active_notes.append(note)
                active_notes[instrument] = new_active_notes
            active_notes[instrument].append(event)
        elif event_type == _NOTE_OFF:
            if sus_active[instrument]:
                pass
            else:
                if event in active_notes[instrument]:
                    active_notes[instrument].remove(event)

    for instrument in active_notes.values():
        for note in instrument:
            note.end = time

    return midi_notes


def get_control_changes(midi, use_drums=True):
    """Retrieves a list of control change events from a given MIDI song.

    Arguments:
        midi (PrettyMIDI): The MIDI song.
    """
    midi_control_changes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        if not midi_instrument.is_drum or use_drums:
            for midi_control_change in midi_instrument.control_changes:
                midi_control_changes.append((
                    midi_instrument.program,
                    num_instrument,
                    midi_instrument.is_drum,
                    midi_control_change
                ))
    return midi_control_changes


def get_notes(midi, use_drums=True):
    """Retrieves a list of MIDI notes (for all instruments) given a MIDI song.

    Arguments:
        midi (PrettyMIDI): The MIDI song.
    """
    midi_notes = []
    for num_instrument, midi_instrument in enumerate(midi.instruments):
        if not midi_instrument.is_drum or use_drums:
            for midi_note in midi_instrument.notes:
                midi_notes.append((
                    midi_instrument.program,
                    num_instrument,
                    midi_instrument.is_drum,
                    midi_note
                ))
    return midi_notes
