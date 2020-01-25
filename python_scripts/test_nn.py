from keras import layers, models, optimizers
from keras.applications.densenet import DenseNet121
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from skimage import img_as_float, io
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dict, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.files = np.array([data_dict[item][0] for item in data_dict.keys()])
        self.labels = [data_dict[item][1] for item in data_dict.keys()]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_dict.keys()) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.files[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_dict.keys()))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = img_as_float(io.imread(ID))
            # Store class
            y[i] = labels_temp[i]

        return X, [y]



with open("/home/simon/PycharmProjects/robert_sql/slide_data_neg1.txt") as fp:
    lines = fp.readlines()
    X_neg = np.zeros((len(lines)-1, len(lines[0].split(" "))))
    params = []
    for i, line in enumerate(lines):
        line_arr = line.replace("\n,", "").split(" ")
        for k, item in enumerate(line_arr):
            if k == 0:
                continue
            elif i == 0:
                params.append(item)
            elif k > 0:
                X_neg[i - 1, k- 1] = float(item)

y_neg = np.zeros(X_neg.shape[0])
lines = None
with open("/home/simon/PycharmProjects/robert_sql/slide_data_pos1.txt") as fp:
    lines = fp.readlines()
    print(len(lines))
    X_pos = np.zeros((len(lines) - 1, len(lines[0].split(" "))))
    y_pos = np.zeros(len(lines))
    for i, line in enumerate(lines):
        line_arr = line.replace("\n,", "").split(" ")
        for k, item in enumerate(line_arr):
            if k == 0:
                continue
            elif i == 0:
                params.append(item)
            elif k > 0:
                X_pos[i - 1, k - 1] = float(item)

y_pos = np.ones(X_pos.shape[0])
X = np.concatenate((X_neg, X_pos))
print(X.shape)
y = np.concatenate((y_neg, y_pos))
print(X.shape, y.shape)

X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 10

params = {'dim': (600,600),
          'batch_size': 5,
          'n_classes': 1,
          'n_channels': 3,
          'shuffle': True}

# data_gen_train = DataGenerator(data_dict, **params)
# data_gen_val = DataGenerator(data_dict_val, **params)

mini_cnn = True
if mini_cnn:
    cnn_model = models.Sequential()
    # cnn_model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(600, 600, 3)))
    # cnn_model.add(layers.MaxPooling2D((2, 2)))
    # cnn_model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    # cnn_model.add(layers.MaxPooling2D((2, 2)))
    # cnn_model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    # cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(300, activation='relu'))
    cnn_model.add(layers.Dense(150, activation='relu'))
    cnn_model.add(layers.Dense(75, activation='relu'))
    cnn_model.add(layers.Dense(15, activation='relu'))
    cnn_model.add(layers.Dense(1, activation='sigmoid'))

else:
    # add a global spatial average pooling layer
    base_model = DenseNet121(include_top=False, weights='imagenet')
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = layers.Dense(200, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = layers.Dense(1, activation='sigmoid')(x)

    cnn_model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    # this is the model we will train

cnn_model.build(input_shape=(10, X_train.shape[1]))
cnn_model.compile(
    optimizer=optimizers.Adam(lr=1e-3),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

cnn_model.summary()
for layer in cnn_model.layers:
    if"bn" in layer.name:
        layer.trainable = False

if not mini_cnn:
    path_checkpoint: str = 'dense_checkpoint.keras'
else:
    path_checkpoint: str = 'mini_checkpoint.keras'

callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)

callback_tensorboard = TensorBoard(log_dir='./22_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callbacks = [callback_checkpoint, callback_tensorboard]

epochs_nr = 4
history = cnn_model.fit(X_train, y_train,
    batch_size=20,
    epochs=epochs_nr,
    validation_data=(X_test, y_test)
)