import pickle
def save_model(file_path, model):
    data = {'model':model}
    with open(file_path,'wb') as f:
        pickle.dump(data,f)

def load_model(file_path):
    with open(file_path,'rb') as f:
        loaded_data = pickle.load(f)
    model = loaded_data['model']
    return model
