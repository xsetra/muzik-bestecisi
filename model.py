import glob
import pickle
import numpy
from termcolor import colored
from music21 import note, chord, instrument, converter
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

class LSTMFactory:

    def __init__(self, sequence_length=100, lstm_size=512, dropout=0.3, dense=256, activation_type='softmax', optimizer='rmsprop', epochs=200, batch_size=64):
        self.notes = []
        self.network_input = []
        self.network_output = []
        self.normalized_input = []
        self.pitchnames = None
        self.note_to_int = None

        self.model = None
        self.sequence_length = sequence_length
        self.lstm_size = lstm_size
        self.dropout = dropout
        self.dense = dense
        self.activation_type = activation_type
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.__n_vocab = 0

    @property
    def n_vocab(self):
        if self.__n_vocab == 0:
            self.__n_vocab = len(set(self.notes))
        return self.__n_vocab

    # Train
    def train_network(self):
        self.read_notes()
        self.prepare_pitches()
        self.prepare_sequences(normalize=False)
        self.create_network(self.network_input)
        self.train()
        self.save_model('test_model.hdf5')

    def save_model(self, file):
        if self.model is None:
            print('[{}] Firstly, train the model'.format(colored('Error', 'yellow')))
            return
        self.model.save_weights(file)
        print('[{}] {}'.format(colored('Save', 'yellow'), file))

    def train(self):
        filepath = 'weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5'
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

    def prepare_pitches(self):
        self.pitchnames = sorted(set(item for item in self.notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    def prepare_sequences(self, normalize=False):
        self.network_input = []
        self.network_output = []
        self.normalized_input = []

        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([self.note_to_int[char] for char in sequence_in])
            self.network_output.append(self.note_to_int[sequence_out])

        n_patterns = len(self.network_input)

        if normalize is False:
            self.network_input = numpy.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
            self.network_input = self.network_input / float(self.n_vocab)
            self.network_output = np_utils.to_categorical(self.network_output)
        else:
            self.normalized_input = numpy.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
            self.normalized_input = self.normalized_input / float(self.n_vocab)

    def read_notes(self):
        self.notes = []
        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file) # Stream
            print("[{}] {}".format(colored('Parsing', 'yellow'), file))

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

        with open('data/notes', 'wb') as filepath:
            pickle.dump(self.notes, filepath)

    # Predict
    def generate(self, weights_file):
        self.prepare_sequences(normalize=True)
        self.create_network(self.normalized_input, weights_file=weights_file=None)
        prediction = self.generate_notes()


    def generate_notes(self):
        start = numpy.random.randint(0, len(self.network_input)-1)

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

        return prediction_output

    def create_midi(prediction_output, outname):
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

        midi_stream.write('midi', fp=outname+".mid")
