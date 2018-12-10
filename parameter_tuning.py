# -*- coding: utf-8 -*-
from generate_model import LSTMTrain, LSTMPredict
from prettytable import PrettyTable
from termcolor import colored
import time


if __name__ == '__main__':
    table = PrettyTable()
    table.field_names = ['Epoch', 'Batch Size', 'Sequence Length', 'Model', 'Music', 'Elapsed']

    tic, toc = None, None

    epochs = [1, 3]
    sequence_length = [10, 100]
    batch_size = [32, 64]

    for epoch in epochs:
        for s_length in sequence_length:
            for b_size in batch_size:
                model_name = "parameter_test/model_E{}_S{}_B{}.hdf5".format(epoch, s_length, b_size)
                trainer = LSTMTrain(model_output=model_name,
                                    epochs=epoch,
                                    batch_size=b_size,
                                    sequence_length=s_length)
                tic = time.time()
                trainer.train()
                toc = time.time()

                predictor = LSTMPredict(model_input=model_name,
                                        epochs=epoch,
                                        batch_size=b_size,
                                        sequence_length=s_length)
                music_name = "parameter_test/music_E{}_S{}_B{}.mid".format(epoch, s_length, b_size)
                predictor.generate(music_name)

                table.add_row([epoch, b_size, s_length, model_name, music_name, toc-tic])

    print(table)
