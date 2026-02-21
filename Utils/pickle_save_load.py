import pickle


def pickle_save(filename, d: dict):
    with open(filename, 'wb') as f:
        # Put them in the file
        pickle.dump(d, f)


def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)