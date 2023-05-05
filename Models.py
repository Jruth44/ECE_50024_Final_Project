import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from GenerateData import GenerateData
from sklearn.linear_model import LinearRegression
import csv

LABEL_COLOR = {-1: "r", 1: "b"}


def read_asc_file(data_folder):
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []

    for i in range(1, 101):
        train_data_file = f"{data_folder}/breast-cancer_train_data_{i}.asc"
        train_labels_file = f"{data_folder}/breast-cancer_train_labels_{i}.asc"
        test_data_file = f"{data_folder}/breast-cancer_test_data_{i}.asc"
        test_labels_file = f"{data_folder}/breast-cancer_test_labels_{i}.asc"

        with open(train_data_file, 'r') as f:
            train_data.extend([list(map(float, line.strip().split())) for line in f.readlines()])

        with open(train_labels_file, 'r') as f:
            train_labels.extend([float(line.strip()) for line in f.readlines()])

        with open(test_data_file, 'r') as f:
            test_data.extend([list(map(float, line.strip().split())) for line in f.readlines()])

        with open(test_labels_file, 'r') as f:
            test_labels.extend([float(line.strip()) for line in f.readlines()])

    # Combine features and labels
    train_data = [[train_labels[i]] + train_data[i] for i in range(len(train_data))]
    test_data = [[test_labels[i]] + test_data[i] for i in range(len(test_data))]

    return train_data, test_data


def load_breast_cancer_data(data_folder):
    train_data, test_data = read_asc_file(data_folder)

    # Splitting the data and labels
    train_labels = [data[0] for data in train_data]
    test_labels = [data[0] for data in test_data]

    return train_data, train_labels, test_data, test_labels


def calculate_loss(true_label, predicted_label):
    return 0 if true_label == predicted_label else 1


def estimate_loss(true_label, noisy_label, p_positive, p_negative, epsilon=1e-8):
    if true_label == noisy_label:
        return ((1 - p_positive) * p_positive + p_negative * p_negative) / (1 - p_positive - p_negative + epsilon)
    else:
        return (p_positive * p_positive + (1 - p_negative) * p_negative) / (1 - p_positive - p_negative + epsilon)


def calculate_accuracy(true_labels, predicted_labels, true_data_map):
    clean_labels = [true_data_map[(d[1], d[2])] for d in true_labels]
    matches = sum(
        1 for clean_label, predicted_label in zip(clean_labels, predicted_labels) if clean_label == predicted_label)
    return round(matches / len(clean_labels), 4)



def find_optimal_alpha(p_positive, p_negative):
    alpha = (1 - p_positive + p_negative) / 2
    return alpha


def alpha_weighted_loss(y_true, y_pred, alpha):
    pos_loss = (1 - alpha) * (y_true == 1) * (y_pred <= 0)
    neg_loss = alpha * (y_true == -1) * (y_pred > 0)
    return pos_loss + neg_loss


def estimate_loss_alpha(x, y, true_label, noisy_label, alpha):
    return alpha * (true_label != noisy_label)


class DataModel:
    def __init__(self, data_size, is_random, p_positive, p_negative, dim=3, ws=(0.5, 0.5)):
        self.data_generator = GenerateData(data_size)
        self.true_data_map = {}
        is_linear = not is_random
        noise_free_data = self.data_generator.generate_data(is_linear)
        self.n1 = sum(1 for d in noise_free_data if d[0] == 1)
        self.n2 = sum(1 for d in noise_free_data if d[0] == -1)
        self.is_random = not is_linear
        self.init_true_data_map(noise_free_data)
        noised_data = self.data_generator.introduce_noise(noise_free_data, p_positive, p_negative)
        self.noised_train_set, self.noised_test_set = self.data_generator.split_data(noised_data)
        self.noisy_test_map = {(x, y): label for label, x, y in self.noised_test_set}
        self.unbiased_loss_pred_map = {}

    def init_true_data_map(self, data):
        for d in data:
            self.true_data_map[(d[1], d[2])] = d[0]

    def train_svm_without_loss_function(self):
        return self.train_svm(self.noised_train_set)

    def train_svm(self, train_set):
        train_X = [(d[1], d[2]) for d in train_set]
        train_y = [d[0] for d in train_set]
        clf = svm.SVC()
        clf.fit(train_X, train_y)
        test_X = [(d[1], d[2]) for d in self.noised_test_set]
        predicted_y = clf.predict(test_X)
        self.unbiased_loss_pred_map = {(xy[0], xy[1]): int(label) for label, xy in zip(predicted_y, test_X)}
        return clf

    def select_classifier_by_kfold(self, p_positive, p_negative, loss_function):
        min_avg_loss = float('inf')
        selected_data = None
        data = np.array(self.noised_train_set)
        kf = KFold(n_splits=5)
        for train_indices, test_indices in kf.split(data):
            train_data = data[train_indices]
            p_negative = sum(1 for d in train_data if d[0] == -1) / len(train_data)
            p_positive = sum(1 for d in train_data if d[0] == 1) / len(train_data)
            avg_losses = [loss_function(self.true_data_map[d[1], d[2]], d[0]) for d in train_data]

            current_avg_loss = np.mean(avg_losses)
            if current_avg_loss < min_avg_loss:
                min_avg_loss = current_avg_loss
                selected_data = train_data
        return self.train_svm(selected_data)

    def comparison_plot(self, noised_test_set, predictions, clf, p_positive, p_negative, model_choice, show_plot=True):
        true_labels = [d[0] for d in noised_test_set]
        noisy_labels = [d[1] for d in noised_test_set]

        x_o = [d[1] for d in self.noised_test_set]
        y_o = [d[2] for d in self.noised_test_set]
        label_o = [self.true_data_map[(x, y)] for x, y in zip(x_o, y_o)]

        f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

        color_o = [LABEL_COLOR[d] for d in label_o]
        ax1.scatter(x_o, y_o, marker='+', c=color_o, s=20, edgecolor='y')
        ax1.set_title('Noise-free')

        x_n = x_o
        y_n = y_o
        label_n = [d[0] for d in self.noised_test_set]
        color_n = [LABEL_COLOR[d] for d in label_n]
        ax2.scatter(x_n, y_n, marker='+', c=color_n, s=20, edgecolor='y')
        ax2.set_title(f'Noise rate: {p_positive} and {p_negative}')

        x_t1 = x_o
        y_t1 = y_o
        noisy_test_X1 = list(zip(x_t1, y_t1))
        if model_choice == "linear_regression":
            pred_label1 = model.predict_linear_regression(clf)
        else:
            pred_label1 = clf.predict(noisy_test_X1)

        label_p1 = [LABEL_COLOR[d] for d in pred_label1]

        matches = sum(
            1 for true_label, predicted_label in zip(label_o, pred_label1) if true_label == predicted_label)
        accuracy = round(matches / len(label_o), 4)

        ax3.scatter(x_t1, y_t1, marker='+', c=label_p1, s=20, edgecolor='y')
        ax3.set_title(f'Accuracy: {accuracy}')

        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        if show_plot:
            plt.show()

        self.save_test_data(f, p_positive, p_negative)
        return accuracy

    def save_test_data(self, f, p_positive, p_negative):
        if self.is_random:
            filename = f"random_test_data_{self.data_generator.data_size}_{p_positive}_{p_negative}.csv"
        else:
            filename = f"linearly_test_data_{self.data_generator.data_size}_{p_positive}_{p_negative}.csv"

        output_file = os.path.join("src", "PredictOutput", filename)

        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['X', 'Y', 'True Label', 'Noisy Label', 'Predicted Label'])
            writer.writerow([f'Noise rate: {p_positive} and {p_negative}'])
            for data in self.noised_test_set:
                x, y = data[1], data[2]
                true_label, noisy_label = self.true_data_map.get((x, y), None), data[0]
                pred_label = self.unbiased_loss_pred_map.get((x, y), None)
                writer.writerow([x, y, true_label, noisy_label, pred_label])

    def train_linear_regression(self):
        train_X = [(d[1], d[2]) for d in self.noised_train_set]
        train_y = [d[0] for d in self.noised_train_set]
        reg = LinearRegression().fit(train_X, train_y)
        return reg

    def predict_linear_regression(self, reg):
        test_X = [(d[1], d[2]) for d in self.noised_test_set]
        predicted_y = reg.predict(test_X)
        return [1 if y >= 0 else -1 for y in predicted_y]


class BreastCancerDataModel(DataModel):
    def __init__(self, data_folder, p_positive, p_negative):
        data_generator = GenerateData(0)  # data_size doesn't matter for UCI data
        train_data, test_data = read_asc_file(data_folder)

        train_data = np.array(train_data)
        test_data = np.array(test_data)

        noised_train_data = data_generator.introduce_noise(train_data, p_positive, p_negative)
        self.true_data_map = {}
        self.noised_train_set = noised_train_data.tolist()
        self.noised_test_set = test_data.tolist()
        self.init_true_data_map(train_data)

        # Initialize the parent class
        super().__init__(data_size=len(train_data) + len(test_data), is_random=False,
                         p_positive=p_positive, p_negative=p_negative, dim=len(train_data[0]))


if __name__ == '__main__':
    data_type = input("Do you want to use linear, random, or UCI dataset? (linear/random/uci) ").lower()

    if data_type == "linear":
        data_size = 4000
        dim = 3
    elif data_type == "random":
        data_size = 5000
        dim = 2
    elif data_type == "uci":
        data_folder = "Health_Data"
    else:
        print("Invalid input. Please enter either 'linear', 'random', or 'uci'")
        sys.exit(1)

    can_continue = input("You've chosen {} data. Do you want to continue? (Y/n) ".format(data_type)).lower()
    if can_continue.startswith("n"):
        sys.exit(0)

    try:
        p_positive = float(input("Enter the noise rate for the positive class (0 to 1): "))
        p_negative = float(input("Enter the noise rate for the negative class (0 to 1): "))
    except ValueError:
        print("Invalid input. Please enter a number between 0 and 1.")
        sys.exit(1)

    if not (0 <= p_positive <= 1) or not (0 <= p_negative <= 1):
        print("Invalid input. Please enter a number between 0 and 1.")
        sys.exit(1)

    if data_type in ("linear", "random"):
        model = DataModel(data_size, is_random=data_type == "random", p_positive=p_positive, p_negative=p_negative,
                          dim=dim)
    elif data_type == "uci":
        model = BreastCancerDataModel(data_folder, p_positive=p_positive, p_negative=p_negative)

    model_choice = input("Choose the model to use: (log_loss/alpha_loss/svm_no_noise/linear_regression) ").lower()

    if model_choice not in ("log_loss", "alpha_loss", "svm_no_noise", "linear_regression"):
        print(
            "Invalid input. Please choose one of the following: log_loss, alpha_loss, svm_no_noise, or linear_regression")
        sys.exit(1)

    if model_choice == "log_loss":
        use_alpha_loss = False
        loss_function = lambda true_label, noisy_label: estimate_loss(
            true_label, noisy_label, p_positive, p_negative)
    elif model_choice == "alpha_loss":
        use_alpha_loss = True
        alpha = find_optimal_alpha(p_positive, p_negative)
        loss_function = lambda true_label, noisy_label: estimate_loss_alpha(
            true_label, noisy_label, p_positive, p_negative, alpha)
    elif model_choice == "svm_no_noise":
        clf = model.train_svm_without_loss_function()
    elif model_choice == "linear_regression":
        clf = model.train_linear_regression()

    if model_choice in ("log_loss", "alpha_loss"):
        clf = model.select_classifier_by_kfold(p_positive, p_negative, loss_function)

    if model_choice == "linear_regression":
        predictions = model.predict_linear_regression(clf)
    else:
        predictions = clf.predict([(d[1], d[2]) for d in model.noised_test_set])

    accuracy = calculate_accuracy(model.noised_test_set, predictions, model.true_data_map)
    print(f"{model_choice} accuracy: {accuracy}")
    model.comparison_plot(model.noised_test_set, predictions, clf, p_positive, p_negative, model_choice, show_plot=True)

