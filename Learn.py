import main
import Neuro
from keras import optimizers
from keras.layers.normalization import layer_normalization

if __name__ == '__main__':
    data = main.Preprocessing()
    # print(data.test_x[0])
    # print(data.test_x_dates[0])
    net = Neuro.NeuroNetwork(data.train_x.shape[1:], filename='76288-fd5-fh5.h5')
    net.fit(data, epoch=100, batch_size=8, optimizer="Adam", validation_split=0.2)
    # net.test(data)
    if '+' == input('save (+/-)'):
        net.save(f"{input('input name model: ')}.h5")


