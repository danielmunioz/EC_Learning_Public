import numpy as np
import pandas as pd
import datetime
from astropy.io import fits

fmt = '%H%M%S'


def auto_find_background(fits_data, amount=0.05):
    data = fits_data
    tmp = (data - np.average(fits_data, 1).reshape(fits_data.shape[0], 1))
    sdevs = np.asarray(np.std(tmp, 0))

    cand = sorted(range(fits_data.shape[1]), key=lambda y: sdevs[y])
    return cand[:max(1, int(amount * len(cand)))]

def auto_const_bg(fits_data):
    realcand = auto_find_background(fits_data)
    bg = np.average(fits_data[:, realcand], 1)
    return bg.reshape(fits_data.shape[0], 1)

def standard_subtract(fits_data):
    return fits_data - auto_const_bg(fits_data)


def length_adjustment(f_length, length, start_position, end_position, window_size):
    """
    Adjust length depending on the relative position of the flare respecting of the file
    """
    true_add = window_size - f_length

    addL = int(true_add / 2)
    addR = np.abs(int(true_add / 2) - true_add)

    new_end = end_position + addR
    new_start = start_position - addL

    if new_end <= length and new_start >= 0:
        return new_start, new_end

    if new_end > length:
        # We're in the right
        new_start = start_position - true_add
        new_end = end_position
        return new_start, new_end

    if new_start < 0:
        # We're in the left
        new_start = start_position
        new_end = new_end + true_add
        return new_start, new_end

    return new_start, new_end


def get_flare(dataframe, index, window_length, bg_subtract=False):

    """
    Extracts Flare from joined Fits file using info from dataframe
    """
    file_here = fits.open(dataframe.loc[index]['remarks'])

    start = datetime.datetime.strptime(dataframe.loc[index]['start'], fmt)
    end = datetime.datetime.strptime(dataframe.loc[index]['end'], fmt)

    time_obs = datetime.datetime.strptime(file_here[0].header['TIME-OBS'][:8], '%H:%M:%S')
    time_end = datetime.datetime.strptime(file_here[0].header['TIME-END'], '%H:%M:%S')

    length = file_here[0].data.shape[1]
    time_window = time_end - time_obs

    # Trying to normalize data
    if time_obs > start:
        sec_toAdd = time_obs - start
        start = start + datetime.timedelta(seconds=sec_toAdd.seconds)

    # with time obs
    start_seconds = start - time_obs
    end_seconds = end - time_obs

    steps_start = int(start_seconds.seconds / time_window.seconds * length)
    steps_end = int(end_seconds.seconds / time_window.seconds * length)
    f_length = steps_end - steps_start

    if f_length < window_length:
        steps_start, steps_end = length_adjustment(f_length, length, steps_start, steps_end, window_length)

    # Getting patch
    
    if bg_subtract:
        flare_patch = standard_subtract(file_here[0].data[:, steps_start:steps_end])
    else:
        flare_patch = file_here[0].data[:, steps_start:steps_end]

    file_here.close()

    return flare_patch


def doubler(data_here):
    """
    Doubles the data over Y axis
    """
    before = np.vstack([data_here[0], data_here[0]])

    for elemen in range(data_here.shape[0] - 1):
        elemen += 1

        here = np.vstack([data_here[elemen], data_here[elemen]])

        before = np.vstack([before, here])

    return before


def stack_window(data, window_length):
    """
    Stacks windows over X-axis
    """
    time_step = window_length
    windows = []

    while time_step <= data.shape[1]:
        x_start = time_step - window_length
        x_end = time_step

        window = data[:, x_start:x_end]
        time_step = time_step + window_length
        windows.append(window)

    return windows


from os import listdir
from os.path import isfile, join


def load_nonFlare(dataSet, window_length, length, is_dir=False):
    """
    Extracts and loads non-flare files.
    """

    main_X = []
    here = 0

    if is_dir:
        onlyfiles = [f for f in listdir(dataSet) if isfile(join(dataSet, f))]
        onlyfiles = np.random.permutation(onlyfiles)

        for index in map(lambda x: dataSet + '\\' + x, onlyfiles):

            data = fits.open(index)[0].data

            if data.shape[0] < 200:
                data = doubler(data)

            slides = stack_window(data, window_length)

            main_X.append(slides)

            here += np.shape(slides)[0]
            if here >= length:
                break

        main_X = np.vstack(main_X)
        main_Y = np.full((len(main_X),), '0')

        return main_X, main_Y

    else:
        for index in dataSet:
            data = fits.open(index)[0].data

            if data.shape[0] < 200:
                data = doubler(data)

            slides = stack_window(data, window_length)

            main_X.append(slides)

            here += np.shape(slides)[0]
            if here >= length:
                break

        main_X = np.vstack(main_X)
        main_Y = np.full((len(main_X),), '0')

        return main_X, main_Y


def load_Flare(dataframe, window_length, bg_subtract=False):

    """
    Loads data from dataframe that CONTAINS FLARES, slicing each element over time by "window_lenght"

    """

    main_X = []

    for index, elemen in dataframe.iterrows():

        flare = get_flare(dataframe, index, window_length, bg_subtract)
        if flare.shape[0] < 200:
            flare = doubler(flare)

        slides = stack_window(flare, window_length)

        main_X.append(slides)

    main_X = np.vstack(main_X)
    main_Y = np.full((len(main_X),), '1')

    return main_X, main_Y


def split_List(directory, percentage):
    """
    Splits List extracted from a folder and, randomly, divides it into Train and eval by percentage

    """
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    onlyfiles = np.random.permutation(onlyfiles)

    train_length = int(np.shape(onlyfiles)[0] * percentage)

    train_set = onlyfiles[0:train_length]
    eval_set = onlyfiles[train_length:np.shape(onlyfiles)[0]]

    train_set = list(map(lambda x: directory + '/' + x, train_set))
    eval_set = list(map(lambda x: directory + '/' + x, eval_set))

    return train_set, eval_set


def split(dataset, percentage):
    """
    Splits dataset and, randomly, divides it into Train and eval by percentage

    """
    train_length = int(dataset.shape[0] * percentage)
    train_set = pd.DataFrame(columns=dataset.columns)
    eval_set = pd.DataFrame(columns=dataset.columns)

    permut = np.random.permutation(dataset.shape[0])
    train_index = permut[0:train_length]
    eval_index = permut[train_length:dataset.shape[0]]

    train_set = train_set.append(dataset.loc[train_index])
    eval_set = eval_set.append(dataset.loc[eval_index])

    return train_set, eval_set
