import pickle

def load_object_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
