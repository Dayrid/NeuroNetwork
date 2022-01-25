import main
import Neuro
from keras import optimizers


if __name__ == '__main__':
    data = main.Preprocessing()
    print(data.test_x[0])
    print(data.test_x_dates[0])
    net = Neuro.NeuroNetwork(data.train_x.shape[1:])
    net.fit(data, epoch=70, batch_size=3, optimizer=optimizers.Adam(), validation_split=0.2)
    net.test(data)
    if '+' == input('save (+/-)'):
        net.save(f"{input('input name model: ')}.h5")


