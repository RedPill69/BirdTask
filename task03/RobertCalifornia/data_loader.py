import numpy as np
import os

def load_challenge():
    result = np.empty((16, 3000, 548))
    for index in range(16):
        result[index, :, :] = np.load(f'../../dataset/challenge/test{index:02}.npy')
    return result

def save_challenge(unique_key, results):
    result_str = ''
    for index in range(results.shape[0]):
        result_str += f'test{index:02},'
        for item in results[index]:
            result_str += f'{item},'
        result_str = result_str[:len(result_str)-1] + '\n'

    with open(f'outputs/{unique_key}.csv', 'w') as f:
        f.write(result_str)

def load_dataset():
    classes = ['comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']
    dataset = {}

    #load files into arrays
    for i, clazz in enumerate(classes):
        dataset[clazz] = {'test_features': [], 'test_labels': [], 'train_features': [], 'train_labels': []}
        folder = f'../../dataset/original/{clazz}'
        np.random.seed(123)
        indices = np.random.choice(np.arange(200), 40)
        for _, __, files in os.walk(folder):
            i = 0
            for file in files:
                if 'labels' in file:
                    continue
                data = np.load(f'{folder}/{file}')
                labels = np.load(f'{folder}/{file.split(".")[0]}.labels.npy')
                if i in indices:
                    dataset[clazz]['test_features'].append(data)
                    dataset[clazz]['test_labels'].append(labels[:, 0])
                else:
                    dataset[clazz]['train_features'].append(data)
                    dataset[clazz]['train_labels'].append(labels[:, 0])
                i += 1
                
    #combine arrays into 1 long array for each bird
    for clazz in classes:
        for group in ['train_features', 'train_labels', 'test_features', 'test_labels' ]:
            dataset[clazz][group] = np.concatenate(dataset[clazz][group], axis=0)

    #extract calls from arrays
    for clazz in classes:
        for group in ['train_', 'test_']:
            f, l = get_calls(dataset[clazz][f'{group}features'], dataset[clazz][f'{group}labels'])
            dataset[clazz][f'{group}features'] = f
            dataset[clazz][f'{group}labels'] = l
    
    return dataset

def get_calls(features, original_labels):
    labels = original_labels.copy()
    labels[labels != 0] = 1
    breaks = np.where(np.diff(labels) != 0)[0]
    centers = np.ceil(np.convolve(breaks+1, np.array([0.5, 0.5]), mode='valid'))[1 if labels[0] == 0 else 0::2].astype(int)
    label_segments = [original_labels[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    feature_segments = [features[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    return feature_segments, label_segments