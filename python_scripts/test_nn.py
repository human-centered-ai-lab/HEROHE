import argparse
from keras import layers, models, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils import shuffle
import pandas as pd
from sqlalchemy import create_engine

DEFAULT_LOG = "submissions_her2/"

## code partially taken from Anna Seranti cnn. model part
## Data gen and other studd done by me
# create sqlalchemy engine

def load_model():
    """create a simple keras model.

        @:return model a keras.model.models instance
    """
    act = "relu"
    model = models.Sequential()
    model.add(layers.Dense(18, activation=act))
    model.add(layers.Dense(36, activation=act)),
    model.add(layers.Dense(100, activation=act))
    model.add(layers.Dense(200, activation=act))
    model.add(layers.Dense(100, activation=act))
    model.add(layers.Dense(36, activation=act))
    model.add(layers.Dense(18, activation=act))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def train(engine_string, log_dir):
    """Load a subset of the nuclei dataset.

        @:param engine_string The read string for the database
        @:param log_dir The directory where the tensorboard files and checkpoints are stored

        @:return None
    """

    engine = create_engine(engine_string)

    # read a table from database into pandas dataframe, replace "tablename" with your table name

    #for check data
    # df = pd.read_sql('SELECT id, name, image_id, her2status, mean_nucleus_area, number_of_nucleus_area_small, number_of_nucleus_area_medium, '
    #                  'number_of_nucleus_area_large, number_of_nucleus_area_extralarge, mean_nucleus_circularity, mean_nucleus_circularity_small, '
    #                  'mean_nucleus_circularity_medium, mean_nucleus_circularity_large, mean_nucleus_circularity_extralarge, mean_nucleus_hematoxylin_od_mean, '
    #                  'nucleus_hematoxylin_od_mean_small, nucleus_hematoxylin_od_mean_medium, nucleus_hematoxylin_od_mean_large, nucleus_hematoxylin_od_mean_extralarge, '
    #                  'aria_circularity_mean, aria_circularity_density_mean, aria_circularity_mean_small, aria_circularity_density_mean_small, aria_circularity_mean_medium, '
    #                  'aria_circularity_density_mean_medium, aria_circularity_mean_large, aria_circularity_density_mean_large, aria_circularity_mean_extralarge, '
    #                  'aria_circularity_density_mean_extralarge, number_of_nucleus_circularity_small, number_of_nucleus_circularity_medium, '
    #                  'number_of_nucleus_circularity_large FROM public.herohe_data WHERE her2status != -1;',engine, index_col='id')

    y = pd.read_sql('SELECT her2status FROM public.herohe_data WHERE her2status != -1;',engine)

    df_train_raw = pd.read_sql('SELECT  mean_nucleus_area, number_of_nucleus_area_small, number_of_nucleus_area_medium, '
                     'number_of_nucleus_area_large, number_of_nucleus_area_extralarge, mean_nucleus_circularity, mean_nucleus_circularity_small, '
                     'mean_nucleus_circularity_medium, mean_nucleus_circularity_large, mean_nucleus_circularity_extralarge, mean_nucleus_hematoxylin_od_mean, '
                     'nucleus_hematoxylin_od_mean_small, nucleus_hematoxylin_od_mean_medium, nucleus_hematoxylin_od_mean_large, nucleus_hematoxylin_od_mean_extralarge, '
                     'aria_circularity_mean, aria_circularity_density_mean, aria_circularity_mean_small, aria_circularity_density_mean_small, aria_circularity_mean_medium, '
                     'aria_circularity_density_mean_medium, aria_circularity_mean_large, aria_circularity_density_mean_large, aria_circularity_mean_extralarge, '
                     'aria_circularity_density_mean_extralarge, number_of_nucleus_circularity_small, number_of_nucleus_circularity_medium, '
                     'number_of_nucleus_circularity_large FROM public.herohe_data WHERE her2status != -1;',engine)


    X = np.array(df_train_raw)
    y = np.array(y)


    # check data
    # y_compare = df["her2status"]
    # for item1, item2 in zip(y, y_compare):
    #     print(item1, "y_new", item2, "y_compare")
    #     input()

    #clean_nans
    X = np.where(np.isnan(X), 0,  X)

    X, y = shuffle(X, y, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    epochs_nr = 1200
    batch_size = 506

    model = load_model()

    model.build(input_shape=(None, X_train.shape[1]))
    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    model.summary()

    callback_checkpoint = ModelCheckpoint(filepath=os.path.join(logdir, path_checkpoint),
                                          verbose=1,
                                          save_weights_only=True)

    callback_tensorboard = TensorBoard(log_dir= log_dir,
                                       histogram_freq=0,
                                       write_graph=False)

    callbacks = [callback_checkpoint, callback_tensorboard]

    history = model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=epochs_nr,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=callbacks
    )


def test(engine_string, chekpoint_path, submit_dir, filename):
    """Load a saved checkpoint for the model and classify the Herohe and create the submission CSV

            @:param engine_string The read string for the database
            @:param log_dir The directory where the tensorboard files and checkpoints are stored

            @:return None
    """

    engine = create_engine(engine_string)
    batch_size = 1

    df_test_raw = pd.read_sql('SELECT mean_nucleus_area, number_of_nucleus_area_small, number_of_nucleus_area_medium, '
                              'number_of_nucleus_area_large, number_of_nucleus_area_extralarge, mean_nucleus_circularity, mean_nucleus_circularity_small, '
                              'mean_nucleus_circularity_medium, mean_nucleus_circularity_large, mean_nucleus_circularity_extralarge, mean_nucleus_hematoxylin_od_mean, '
                              'nucleus_hematoxylin_od_mean_small, nucleus_hematoxylin_od_mean_medium, nucleus_hematoxylin_od_mean_large, nucleus_hematoxylin_od_mean_extralarge, '
                              'aria_circularity_mean, aria_circularity_density_mean, aria_circularity_mean_small, aria_circularity_density_mean_small, aria_circularity_mean_medium, '
                              'aria_circularity_density_mean_medium, aria_circularity_mean_large, aria_circularity_density_mean_large, aria_circularity_mean_extralarge, '
                              'aria_circularity_density_mean_extralarge, number_of_nucleus_circularity_small, number_of_nucleus_circularity_medium, '
                              'number_of_nucleus_circularity_large FROM public.herohe_data WHERE her2status != -1  OR her2status IS NULL;',
                              engine)

    files_test_raw = np.array(pd.read_sql('SELECT name FROM public.herohe_data WHERE her2status = -1  OR her2status IS NULL;', engine))

    X_test = np.array(df_test_raw)
    X_test = np.where(np.isnan(X_test), 0, X_test)

    print(files_test_raw.shape, X_test.shape)

    model = load_model()
    model.build(input_shape=(batch_size, X_test.shape[1]))
    model.load_weights(chekpoint_path)
    model.summary()
    test_results = {}
    for item1, item2 in zip(X_test, files_test_raw):
        test_results.update({item2[0]: []})

        sample = np.reshape(item1, (batch_size, X_test.shape[1]))
        result = model.predict(sample, batch_size=1)

        test_results[item2[0]].append(result.squeeze())
        if result.squeeze() < .5:
            test_results[item2[0]].append(0)
        else:
            test_results[item2[0]].append(1)

    # get submission path
    if submit_dir == DEFAULT_LOG:
        path1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), submit_dir)
        if not os.path.exists(path1):
            os.mkdir(path1)
        path1 = os.path.join(path1, filename)
    else:
        if not os.path.exists(submit_dir):
            os.mkdir(submit_dir)
        path1 = os.path.join(submit_dir, filename)

    csvfile = open(path1, 'w')
    features = []
    features.insert(0, 'hard_preditction')
    features.insert(0, 'soft_prediction')
    features.insert(0, 'caseID')

    line = ','.join(map(str, features))
    csvfile.write(line + "\n")
    for item in test_results:
        line_arr = []
        line_arr.append(item.split("_")[0])
        for item2 in test_results[item]:
            line_arr.append(item2)
        line = ','.join(map(str, line_arr))
        csvfile.write(line + "\n")
    csvfile.close()


################################### main fucntion #############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Herohe challenge Test/Train Skript')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--logs', required=False,
                        default="logs_her2/",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory path (path/to/logdir)')
    parser.add_argument('--subdir', required=False,
                        default = DEFAULT_LOG,
                        metavar="/path/to/subdir/",
                        help="path to the subdirectory to save submission file to /path/to/subdir/")
    parser.add_argument('--filename', required=False,
                        default = "Her2_test_results.csv",
                        metavar="<filename>",
                        help="The filname of the generated CSV file")

    args = parser.parse_args()

    DATABASES = {
        'challenge': {
            'NAME': 'postgres',
            'USER': 'postgres',
            'PASSWORD': 'postgres',
            'HOST': '127.0.0.1',
            'PORT': 5432,
        },
    }

    logdir = args.logs
    path_checkpoint: str = 'mini_checkpoint.keras'
    chekpoint_path = os.path.join(logdir, path_checkpoint)


    # choose the database to use
    db = DATABASES['challenge']
    np.random.seed(42)

    # construct an engine connection string
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=db['USER'],
        password=db['PASSWORD'],
        host=db['HOST'],
        port=db['PORT'],
        database=db['NAME'],
    )

    if args.command == "train":
        train(engine_string, args.logs)
    elif args.command == "test":
        test(engine_string, chekpoint_path, args.subdir, args.filename)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
