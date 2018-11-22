import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from os.path import exists

class LSTMBase:

    def __init__(self, sequence_length=100, lstm_size=512, dropout=0.3, dense=256, activation_type='softmax', optimizer='rmsprop', epochs=200, batch_size=64):
        self.notes_path = "data/notes"

        self.model = None
        self.notes = []
        self.sequence_length = sequence_length
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.dense = dense
        self.activation_type = activation_type
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.pitchnames = None
        self.note_to_int = None
        self.int_to_note = None

        self.get_notes()

    @property
    def n_vocab(self):
        return len(set(self.notes))

    def get_notes(self):
        if exists(self.notes_path):
            with open(self.notes_path, "rb") as fileobj:
                self.notes = pickle.load(fileobj)
                return

        self.notes = []
        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)
            print("{} {}".format("[Parsing]", file))

            notes_to_parse = None
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    self.notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.notes.append('.'.join(str(n) for n in element.normalOrder))

        with open(self.notes_path, 'wb') as fileobj:
            pickle.dump(self.notes, fileobj)

    def prepare_sequences(self):
        pass

    def create_network(self, inputs, weights_file=None):
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_size,
                            input_shape=(inputs.shape[1],
                                         inputs.shape[2]),
                            return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(self.lstm_size,
                            return_sequences=True))
        self.model.add(Dropout(self.dropout))
        self.model.add(LSTM(self.lstm_size))
        self.model.add(Dense(self.dense))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.n_vocab))
        self.model.add(Activation(self.activation_type))
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

        if weights_file is not None:
            self.model.load_weights(weights_file)


class LSTMTrain(LSTMBase):
    def __init__(self, model_output, *args, **kwargs):
        LSTMBase.__init__(self, *args, **kwargs)
        self.network_input = None
        self.network_output = None
        self.model_output = model_output

        self.prepare_sequences()
        self.create_network(self.network_input)

    def prepare_sequences(self):
        self.pitchnames = sorted(set(item for item in self.notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))

        self.network_input = []
        self.network_output = []

        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([self.note_to_int[char] for char in sequence_in])
            self.network_output.append(self.note_to_int[sequence_out])

        n_patterns = len(self.network_input)
        self.network_input = numpy.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
        self.network_input = self.network_input / float(self.n_vocab)
        self.network_output = np_utils.to_categorical(self.network_output)

    def train(self):
        filepath = 'modelcheckpoints/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
        checkpoint = ModelCheckpoint(filepath,
                                     monitor='loss',
                                     verbose=0,
                                     save_best_only=True,
                                     mode='min')
        callbacks_list = [checkpoint]
        self.model.fit(self.network_input,
                       self.network_output,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       callbacks=callbacks_list)
        self.model.save(self.model_output)

class LSTMPredict(LSTMBase):
    def __init__(self, model_input, *args, **kwargs):
        LSTMBase.__init__(self, *args, **kwargs)

        self.network_input = None
        self.normalized_input = None
        self.output = None

        self.prepare_sequences()
        self.create_network(self.normalized_input, weights_file=model_input)

    def prepare_sequences(self):
        self.pitchnames = sorted(set(item for item in self.notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))

        self.network_input = []
        self.output = []
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([self.note_to_int[char] for char in sequence_in])
            self.output.append(self.note_to_int[sequence_out])

        n_patterns = len(self.network_input)
        self.normalized_input = numpy.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
        self.normalized_input = self.normalized_input / float(self.n_vocab)

    def generate(self, midi_file):
        start = numpy.random.randint(0, len(self.network_input)-1)

        self.int_to_note = dict((number, note) for number, note in enumerate(self.pitchnames))

        pattern = self.network_input[start]
        prediction_output = []

        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)

            prediction = self.model.predict(prediction_input, verbose=0)

            index = numpy.argmax(prediction)
            result = self.int_to_note[index]
            prediction_output.append(result)

            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        self.create_midi(prediction_output, midi_file)

    def create_midi(self, prediction_output, midi_file):
        """ convert the output from the prediction to notes and create a midi file
            from the notes """
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 0.5
        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=midi_file)

if __name__ == '__main__':
    trainer = LSTMTrain(model_output="model_apo_test.hdf5", epochs=10, batch_size=8, sequence_length=10)
    trainer.train()

    generator = LSTMPredict(model_input="model_apo_test.hdf5", epochs=10, batch_size=8, sequence_length=10)
    generator.generate("test_apo3.mid")
