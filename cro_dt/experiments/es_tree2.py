from cro_dt.sup_configs import get_config
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from cro_dt.cythonfns.TreeEvaluation import get_leaf_simple


class TreeFlat:
    def __init__(self, config, depth, attributes, thresholds, labels):
        self.config = config
        self.depth = depth
        self.attributes = attributes
        self.thresholds = thresholds
        self.labels = labels

    def copy(self):
        return TreeFlat(self.config, self.depth, self.attributes.copy(), self.thresholds.copy(), self.labels.copy())

    def evaluate(self, X, y):
        pred = [self.predict(x) for x in X]
        return np.mean(pred == y)

    def predict(self, x):
        return self.labels[get_leaf_simple(x, self.attributes, self.thresholds, self.depth)]
        # return self.labels[self.get_leaf(x)]

    def get_leaf(self, x):
        curr_depth = self.depth - 1
        node_idx = 0
        leaf_idx = 0

        while curr_depth >= 0:
            if x[self.attributes[node_idx]] <= self.thresholds[node_idx]:
                node_idx += 1
            else:
                node_idx += 2 ** curr_depth
                leaf_idx += 2 ** curr_depth
            curr_depth -= 1

        return leaf_idx

    def update_labels(self, X, y):
        pred_leaves_idx = [self.get_leaf(x) for x in X]
        pred_leaves_labels = [[y[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(2 ** self.depth)]

        for i in range(2 ** self.depth):
            if len(pred_leaves_labels[i]) == 0:
                self.labels[i] = np.argmax(np.bincount(y))
            else:
                self.labels[i] = np.argmax(np.bincount(pred_leaves_labels[i]))

    def mutate(self):
        operator = np.random.choice(["attribute", "threshold"])
        if operator == "attribute":
            self._mutate_attribute()
        elif operator == "threshold":
            self._mutate_threshold()

    def _mutate_attribute(self):
        node_idx = np.random.randint(2 ** self.depth - 1)
        self.attributes[node_idx] = np.random.randint(self.config['n_attributes'])
        self.thresholds[node_idx] = np.random.uniform(self.config['attributes_metadata'][self.attributes[node_idx]][0],
                                                      self.config['attributes_metadata'][self.attributes[node_idx]][1])

    def _mutate_threshold(self):
        node_idx = np.random.randint(2 ** self.depth - 1)
        self.thresholds[node_idx] = np.random.uniform(self.config['attributes_metadata'][self.attributes[node_idx]][0],
                                                      self.config['attributes_metadata'][self.attributes[node_idx]][1])

    @staticmethod
    def generate_random(config, depth):
        attributes = []
        thresholds = []
        labels = []
        for i in range(2 ** depth):
            attributes.append(np.random.randint(config['n_attributes']))
            thresholds.append(np.random.uniform(config['attributes_metadata'][attributes[-1]][0],
                                                config['attributes_metadata'][attributes[-1]][1]))
            labels.append(0)

        attributes = np.array(attributes, dtype=np.int64)
        thresholds = np.array(thresholds, dtype=np.float64)
        labels = np.array(labels, dtype=np.int64)

        return TreeFlat(config, depth, attributes, thresholds, labels)

    def __str__(self):
        stack = [(0, 0, self.depth - 1)]
        output = ""

        while len(stack) > 0:
            node_id, leaf_id, depth = stack.pop()
            output += "-" * (self.depth - depth) + " "

            if depth == -1:
                output += str(self.labels[leaf_id])
            else:
                output += self.config['attributes'][self.attributes[node_id]]
                output += " <= "
                output += '{:.5f}'.format(self.thresholds[node_id])

                stack.append((node_id + 1, leaf_id, depth - 1))
                stack.append((node_id + 2 ** depth, leaf_id + 2 ** depth, depth - 1))
            output += "\n"

        return output


if __name__ == "__main__":
    dataset = "transfusion"

    config = get_config(dataset)
    df = pd.read_csv(f"cro_dt/experiments/data/{dataset}.csv")
    feat_names = df.columns[:-1]
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)

    config["attributes_metadata"] = [(np.min(X_i), np.max(X_i)) for X_i in np.transpose(X_train.astype(np.float32))]
    tree = TreeFlat.generate_random(config, 2)
    print(tree)
    tree.mutate()
    tree.update_labels(X_train, y_train)
    print("\nMutated!\n")
    print(tree)
    tree.mutate()
    tree.update_labels(X_train, y_train)
    print("\nMutated!\n")
    print(tree)
    tree.mutate()
    tree.update_labels(X_train, y_train)
    print("\nMutated!\n")
    print(tree)
