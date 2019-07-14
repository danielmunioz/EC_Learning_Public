import numpy as np
import pandas as pd
import datetime
from astropy.io import fits

fmt = '%H%M%S'


def next_bin_step(bin_step):
    if bin_step <=3:
        bin_step+=1
        return bin_step
    else:
        return -1


def get_window(data, window_height, window_lenght, bin_step):
    time_step = window_lenght
    windows = []

    bin_controller = next_bin_step(bin_step)

    bin_start = (bin_controller * window_height) - window_height
    bin_end = bin_controller * window_height

    while time_step <= data.shape[1]:
        x_start = time_step - window_lenght
        x_end = time_step

        window = data[bin_start:bin_end, x_start:x_end]
        time_step = time_step + window_lenght
        windows.append(window)

    return windows


def stack_patches(data, window_height, window_lenght):
    step = int(data.shape[0] / window_height - 1)

    frames = get_window(data, window_height, window_lenght, step)
    frame_set = np.array(frames)
    frame_set = np.flip(frame_set, 0)
    step -= 1

    while step != -1:
        frames = get_window(data, window_height, window_lenght, step)
        frames = np.flip(frames, 0)
        frame_set = np.vstack((frame_set, frames))
        step -= 1

    frame_set = np.flip(frame_set, 0)
    return frame_set


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


def length_adjustment(f_lenght, lenght, start_position, end_position, window_size):
    true_add = window_size - f_lenght

    addL = int(true_add / 2)
    addR = np.abs(int(true_add / 2) - true_add)

    new_end = end_position + addR
    new_start = start_position - addL

    if new_end <= lenght and new_start >= 0:
        return new_start, new_end

    if new_end > lenght:
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
    file_here = fits.open(dataframe.loc[index]['remarks'])

    start = datetime.datetime.strptime(dataframe.loc[index]['start'], fmt)
    end = datetime.datetime.strptime(dataframe.loc[index]['end'], fmt)

    time_obs = datetime.datetime.strptime(file_here[0].header['TIME-OBS'][:8], '%H:%M:%S')
    time_end = datetime.datetime.strptime(file_here[0].header['TIME-END'], '%H:%M:%S')

    lenght = file_here[0].data.shape[1]
    time_window = time_end - time_obs

    # Trying to normalize data
    if time_obs > start:
        sec_toAdd = time_obs - start
        start = start + datetime.timedelta(seconds=sec_toAdd.seconds)

    # with time obs
    start_seconds = start - time_obs
    end_seconds = end - time_obs

    steps_start = int(start_seconds.seconds / time_window.seconds * lenght)
    steps_end = int(end_seconds.seconds / time_window.seconds * lenght)
    f_length = steps_end - steps_start

    if f_length < window_length:
        steps_start, steps_end = length_adjustment(f_length, lenght, steps_start, steps_end, window_length)

    # Getting patch
    
    if bg_subtract:
        flare_patch = standard_subtract(file_here[0].data[:, steps_start:steps_end])
    else:
        flare_patch = file_here[0].data[:, steps_start:steps_end]

    file_here.close()

    return flare_patch


def data_loader(dataframe, window_height, window_length, bg_subtract=False):

    """
    It was originally made to load data slicing the data over Time-Frequency bins, to load data slicing only over time
    use slice_overTime

    :param dataframe:
    :param window_height:
    :param window_length:
    :param bg_subtract:
    :return:
    """
    main_X = []
    main_Y = []

    for index, elemen in dataframe.iterrows():

        flare = get_flare(dataframe, index, window_length, bg_subtract)
        patchs_temp = stack_patches(flare, window_height, window_length)

        Y = np.full((len(patchs_temp),), dataframe.loc[index]['class'])

        main_Y = np.append(main_Y, Y)
        main_X.append(patchs_temp)

    main_X = np.vstack(main_X)

    total_examples = dataframe.groupby('class').size()

    classes = np.array(total_examples.index)

    return main_X, main_Y, classes


def doubler(data_here):
    before = np.vstack([data_here[0], data_here[0]])

    for elemen in range(data_here.shape[0] - 1):
        elemen += 1

        here = np.vstack([data_here[elemen], data_here[elemen]])

        before = np.vstack([before, here])

    return before


def stack_window(data, window_lenght):
    time_step = window_lenght
    windows = []

    while time_step <= data.shape[1]:
        x_start = time_step - window_lenght
        x_end = time_step

        window = data[:, x_start:x_end]
        time_step = time_step + window_lenght
        windows.append(window)

    return windows


def slice_overTime(dataframe, window_length, bg_subtract=False):
    """
    Use to load data slicing ONLY over Time axis.

    :param dataframe:
    :param window_length:
    :param bg_subtract:
    :return: data, labels and classes
    """

    main_X = []
    main_Y = []

    for index, elemen in dataframe.iterrows():

        flare = get_flare(dataframe, index, window_length, bg_subtract)
        if flare.shape[0] < 200:
            flare = doubler(flare)

        slides = stack_window(flare, window_length)

        # getting labels for Y
        Y = np.full((len(slides),), dataframe.loc[index]['class'])

        # appending
        main_Y = np.append(main_Y, Y)
        main_X.append(slides)

    main_X = np.vstack(main_X)

    total_examples = dataframe.groupby('class').size()

    classes = np.array(total_examples.index)

    return main_X, main_Y, classes


from os import listdir
from os.path import isfile, join


def load_nonFlare(dataSet, window_length, length, is_dir=False):
    """
    If normalize, window_height should be 200 (BC is the max value found so far) so we set the frequency bin to 200

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
    If normalize, window_height should be 200 (BC is the max value found so far) so we set the frequency bin to 200
    IF using already subtracted data have False by default

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

    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    onlyfiles = np.random.permutation(onlyfiles)

    train_length = int(np.shape(onlyfiles)[0] * percentage)

    train_set = onlyfiles[0:train_length]
    eval_set = onlyfiles[train_length:np.shape(onlyfiles)[0]]

    train_set = list(map(lambda x: directory + '\\' + x, train_set))
    eval_set = list(map(lambda x: directory + '\\' + x, eval_set))

    return train_set, eval_set


def split(dataset, percentage):
    train_length = int(dataset.shape[0] * percentage)
    train_set = pd.DataFrame(columns=dataset.columns)
    eval_set = pd.DataFrame(columns=dataset.columns)

    permut = np.random.permutation(dataset.shape[0])
    train_index = permut[0:train_length]
    eval_index = permut[train_length:dataset.shape[0]]

    train_set = train_set.append(dataset.loc[train_index])
    eval_set = eval_set.append(dataset.loc[eval_index])

    return train_set, eval_set
