import numpy as np
import os

def load_challenge():
    result = np.empty((16, 3000, 548))
    for index in range(16):
        result[index, :, :] = np.load(f'../dataset/challenge/test{index:02}.npy')
    return result

def save_challenge(unique_key, results):
    result_str = ''
    for index in range(results.shape[0]):
        result_str += f'test{index:02},'
        for item in results[index]:
            result_str += f'{item},'
        result_str = result_str[:len(result_str)-1] + '\n'

    with open(f'{unique_key}.csv', 'w') as f:
        f.write(result_str)

def load_dataset():
    classes = ['comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']
    dataset = {}

    #load files into arrays
    for clazz in classes:
        dataset[clazz] = {'features': [], 'labels': []}
        folder = f'../dataset/original/{clazz}'
        for _, __, files in os.walk(folder):
            for file in files:
                if 'labels' in file:
                    continue
                data = np.load(f'{folder}/{file}')
                labels = np.load(f'{folder}/{file.split(".")[0]}.labels.npy')
                dataset[clazz]['features'].append(data)
                dataset[clazz]['labels'].append(labels[:, 0])
            
    #combine arrays into 1 long array for each bird
    for clazz in classes:
        dataset[clazz]['features'] = np.concatenate(dataset[clazz]['features'], axis=0)
        dataset[clazz]['labels'] = np.concatenate(dataset[clazz]['labels'], axis=0)

    #extract calls from arrays
    for clazz in classes:
        f, l = get_calls(dataset[clazz]['features'], dataset[clazz]['labels'])
        dataset[clazz]['features'] = f
        dataset[clazz]['labels'] = l
    
    #split into train and test set
    for clazz in classes:
        f = dataset[clazz]['features']
        l = dataset[clazz]['labels']

        dataset[clazz]['train_features'] = []
        dataset[clazz]['train_labels'] = []

        dataset[clazz]['test_features'] = []
        dataset[clazz]['test_labels'] = []

        indices = np.arange(len(l))
        train_index = np.random.choice(indices, size=int(len(l) * 0.8), replace=False)
        for i in indices:
            if i in train_index:
                dataset[clazz]['train_features'].append(dataset[clazz]['features'][i])
                dataset[clazz]['train_labels'].append(dataset[clazz]['labels'][i])
            else:
                dataset[clazz]['test_features'].append(dataset[clazz]['features'][i])
                dataset[clazz]['test_labels'].append(dataset[clazz]['labels'][i])
    
    return dataset

def get_calls(features, original_labels):
    labels = original_labels.copy()
    labels[labels != 0] = 1
    breaks = np.where(np.diff(labels) != 0)[0]
    centers = np.ceil(np.convolve(breaks+1, np.array([0.5, 0.5]), mode='valid'))[1 if labels[0] == 0 else 0::2].astype(int)
    label_segments = [original_labels[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    feature_segments = [features[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    return feature_segments, label_segments