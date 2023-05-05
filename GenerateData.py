import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class GenerateData:
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.data_size = n_samples

    def create_base_data(self, ratio=0.5):
        data = []
        while len(data) < self.n_samples:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            label = -1 if x - 0.6 * 100 >= y else 1
            if (label == 1 and len([d for d in data if d[0] == 1]) < self.n_samples * ratio) or (
                    label == -1 and len([d for d in data if d[0] == -1]) < self.n_samples * (1 - ratio)):
                data.append([label, x, y])
        return data

    def generate_data(self, is_linear):
        if is_linear:
            return np.array(self.create_base_data())
        else:
            return np.array(self.generate_random_data())

    def generate_random_data(self, num_features=3, class_weights=(0.5, 0.5)):
        X, y = make_classification(n_samples=self.n_samples, n_features=num_features, n_redundant=1, n_informative=2,
                                   n_clusters_per_class=2, flip_y=0.0001, weights=class_weights, random_state=1)
        y = np.where(y == 0, -1, y)
        data = np.column_stack((y, X[:, :2]))
        return data

    def introduce_noise(self, data, noise_ratio1, noise_ratio2):
        data_copy = data.copy()
        idx_class1 = np.where(data_copy[:, 0] == 1)[0]
        idx_class2 = np.where(data_copy[:, 0] == -1)[0]

        num_flip1 = int(len(idx_class1) * noise_ratio1)
        num_flip2 = int(len(idx_class2) * noise_ratio2)

        idx_flip1 = np.random.choice(idx_class1, num_flip1, replace=False)
        idx_flip2 = np.random.choice(idx_class2, num_flip2, replace=False)

        data_copy[idx_flip1, 0] = -1
        data_copy[idx_flip2, 0] = 1

        return data_copy

    def split_data(self, data):
        X_train, X_test = train_test_split(data, shuffle=True)
        print(f"Data split into training and testing sets: {len(X_train)} vs {len(X_test)}")
        return X_train, X_test

    def plot_data(self, all_data, train_data, test_data):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

        ax1.scatter(all_data[:, 1], all_data[:, 2], marker='+', c=all_data[:, 0], s=20, edgecolor='k')
        ax1.set_title('All Data')

        ax2.scatter(train_data[:, 1], train_data[:, 2], marker='+', c=train_data[:, 0], s=20, edgecolor='k')
        ax2.set_title('Training Data')

        ax3.scatter(test_data[:, 1], test_data[:, 2], marker='+', c=test_data[:, 0], s=20, edgecolor='k')
        ax3.set_title('Testing Data')

        plt.show()


def main():
    data_size = 5000
    noise_ratio1, noise_ratio2 = 0.2, 0.2

    generator = GenerateData(data_size)

    data = np.array(generator.create_base_data())
    noise_free_data = np.array(generator.generate_random_data())
    noisy_data = generator.introduce_noise(noise_free_data, noise_ratio1, noise_ratio2)

    train_data, test_data = generator.split_data(noisy_data)

    generator.plot_data(noise_free_data, train_data, test_data)

    # Save the data as CSV files
    pd.DataFrame(noise_free_data).to_csv("noise_free_data.csv", header=False, index=False)
    pd.DataFrame(noisy_data).to_csv("noisy_data.csv", header=False, index=False)
    pd.DataFrame(train_data).to_csv("train_data.csv", header=False, index=False)
    pd.DataFrame(test_data).to_csv("test_data.csv", header=False, index=False)


if __name__ == "__main__":
    main()


