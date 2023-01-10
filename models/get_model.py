from models.bilstm import BiLSTM

def get_model(key, *args, **kwargs):
    if key.lower() == "bilstm":
        return BiLSTM(*args, **kwargs)