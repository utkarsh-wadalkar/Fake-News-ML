import pickle

with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)

print("Keys in model.pkl:", model_data.keys())
