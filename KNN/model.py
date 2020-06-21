import numpy as np

from collections import Counter

class KNearestNeighbor:

    def __init__(self, k, x_train, y_train, dist='l2'):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train
        assert self.k <= x_train.shape[0]


        if dist == 'l1':
            self.func = self.l1_norm
        elif dist == 'l2':
            self.func = self.l2_norm

    def fit(self, x_test):

        pred = []
        for i in range(x_test.shape[0]):

            distance = self.func(self.x_train, x_test[i])

            sorted_train_indices = np.argsort(distance, axis=0)

            sorted_k_labels = []
            labels = []
            for j in range(x_test.shape[1]):
                sort_idx = sorted_train_indices[:, j]
                sorted_label = self.y_train[:, j]
                sorted_k_labels.append(sorted_label[sort_idx][:self.k])
                     
                label = Counter(sorted_k_labels[j]).most_common(1)[0][0]
                labels.append(label)
            pred.append(labels)

        pred = np.asarray(pred).astype(np.int64)

        return pred

    def l1_norm(self, x, y):
        l1 = np.abs(x - y)
        return l1

    def l2_norm(self, x, y):
        l2 = np.square(x - y)
        return l2
    

def debug():
    
    np.random.seed(seed=0)
    x_train = np.random.randint(0, 100, (60, 60000)).astype(np.float32)
    y_train = np.random.randint(0, 2, (60, 60000)).astype(np.int64)

    print(x_train)
    print(y_train)

    model = KNearestNeighbor(3, x_train, y_train)

    x_valid = np.random.randint(0, 100, (10, 60000)).astype(np.float32)

    pred = model.fit(x_valid)
    print(pred)


if __name__ == '__main__':
    debug()