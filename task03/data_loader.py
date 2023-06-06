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

    with open(f'outputs/{unique_key}.csv', 'w') as f:
        f.write(result_str)

def load_dataset():
    classes = ['comcuc', 'cowpig1', 'eucdov', 'eueowl1', 'grswoo', 'tawowl1']
    dataset = {}

    #load files into arrays
    for i, clazz in enumerate(classes):
        dataset[clazz] = {'features': [], 'labels': [], 'agreement': []}
        folder = f'../dataset/original/{clazz}'
        for _, __, files in os.walk(folder):
            for file in files:
                if 'labels' in file:
                    continue
                data = np.load(f'{folder}/{file}')
                labels = np.load(f'{folder}/{file.split(".")[0]}.labels.npy')
                dataset[clazz]['features'].append(data)
                dataset[clazz]['labels'].append(labels[:, 0])
                agreements = np.empty((len(labels), 2))
                agreements[labels[:, 0] != 0, 0] = np.count_nonzero(labels[labels[:, 0] != 0][:, 1:], axis=1) / labels.shape[1]
                agreements[labels[:, 0] == 0, 0] = (labels.shape[1] - np.count_nonzero(labels[labels[:, 0] == 0][:, 1:], axis=1)) / labels.shape[1]
                agreements[:, 1] = i+1
                dataset[clazz]['agreement'].append(agreements)
            
    #combine arrays into 1 long array for each bird
    for clazz in classes:
        dataset[clazz]['features'] = np.concatenate(dataset[clazz]['features'], axis=0)
        dataset[clazz]['labels'] = np.concatenate(dataset[clazz]['labels'], axis=0)
        dataset[clazz]['agreement'] = np.concatenate(dataset[clazz]['agreement'], axis=0)

    #extract calls from arrays
    for clazz in classes:
        f, l, a = get_calls(dataset[clazz]['features'], dataset[clazz]['labels'], dataset[clazz]['agreement'])
        dataset[clazz]['features'] = f
        dataset[clazz]['labels'] = l
        dataset[clazz]['agreement'] = a
    
    #split into train and test set
    for clazz in classes:

        dataset[clazz]['train_features'] = []
        dataset[clazz]['train_labels'] = []
        dataset[clazz]['train_agreement'] = []

        dataset[clazz]['test_features'] = []
        dataset[clazz]['test_labels'] = []
        dataset[clazz]['test_agreement'] = []

        l = dataset[clazz]['labels']
        indices = np.arange(len(l))
        np.random.seed(123)
        train_index = np.random.choice(indices, size=int(len(l) * 0.8), replace=False)
        for i in indices:
            if i in train_index:
                dataset[clazz]['train_features'].append(dataset[clazz]['features'][i])
                dataset[clazz]['train_labels'].append(dataset[clazz]['labels'][i])
                dataset[clazz]['train_agreement'].append(dataset[clazz]['agreement'][i])
            else:
                dataset[clazz]['test_features'].append(dataset[clazz]['features'][i])
                dataset[clazz]['test_labels'].append(dataset[clazz]['labels'][i])
                dataset[clazz]['test_agreement'].append(dataset[clazz]['agreement'][i])
    
    return dataset

def get_calls(features, original_labels, agreements):
    labels = original_labels.copy()
    labels[labels != 0] = 1
    breaks = np.where(np.diff(labels) != 0)[0]
    centers = np.ceil(np.convolve(breaks+1, np.array([0.5, 0.5]), mode='valid'))[1 if labels[0] == 0 else 0::2].astype(int)
    label_segments = [original_labels[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    feature_segments = [features[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    agreement_segments = [agreements[centers[i]:centers[i+1]] for i in range(len(centers)-1)]
    return feature_segments, label_segments, agreement_segments