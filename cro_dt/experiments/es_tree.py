from cro_dt.sup_configs import get_config
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Tree:
    def __init__(self, config, attribute, threshold, label, left, right, fitness=None, id=None):
        self.config = config
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.left = left
        self.right = right
        self.id = id
        self.fitness = fitness

    def predict(self, state):
        return self.get_leaf(state).label

    def get_leaf(self, state):
        if self.is_leaf():
            return self

        if state[self.attribute] <= self.threshold:
            return self.left.get_leaf(state)
        else:
            return self.right.get_leaf(state)

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_inner_nodes(self):
        if self.is_leaf():
            return []
        else:
            return [self] + self.left.get_inner_nodes() + self.right.get_inner_nodes()

    def get_leaf_nodes(self):
        if self.is_leaf():
            return [self]
        else:
            return self.left.get_leaf_nodes() + self.right.get_leaf_nodes()

    def update_labels(self, X, y):
        leaves = self.get_leaf_nodes()
        pred_leaves_idx = [self.get_leaf(x).id for x in X]
        pred_leaves_labels = [[y[i] for i in range(len(y)) if pred_leaves_idx[i] == j] for j in range(len(leaves))]

        for i, leaf in enumerate(leaves):
            if len(pred_leaves_labels[i]) == 0:
                leaf.label = np.argmax(np.bincount(y))
            else:
                leaf.label = np.argmax(np.bincount(pred_leaves_labels[i]))

    def evaluate(self, X, y):
        pred = [self.predict(x) for x in X]
        return np.mean(pred == y)

    def mutate(self):
        operator = np.random.choice(["attribute", "threshold"])
        if operator == "attribute":
            return self._mutate_attribute()
        elif operator == "threshold":
            return self._mutate_threshold()

    def _mutate_attribute(self):
        node = np.random.choice(self.get_inner_nodes())
        node.attribute = np.random.randint(self.config['n_attributes'])
        node.threshold = np.random.uniform(self.config['attributes_metadata'][node.attribute][0],
                                           self.config['attributes_metadata'][node.attribute][1])
        return self

    def _mutate_threshold(self):
        node = np.random.choice(self.get_inner_nodes())
        node.threshold = np.random.uniform(self.config['attributes_metadata'][node.attribute][0],
                                           self.config['attributes_metadata'][node.attribute][1])
        return self

    @staticmethod
    def generate_random_tree(config, depth):
        if depth == 0:
            return Tree(None, None, None, -1, None, None)

        attribute = np.random.randint(config['n_attributes'])
        threshold = np.random.uniform(config['attributes_metadata'][attribute][0],
                                      config['attributes_metadata'][attribute][1])
        left = Tree.generate_random_tree(config, depth - 1)
        right = Tree.generate_random_tree(config, depth - 1)

        root = Tree(config, attribute, threshold, None, left, right)

        for i, leaf in enumerate(root.get_leaf_nodes()):
            leaf.id = i

        return root

    def copy(self):
        return Tree(self.config, self.attribute, self.threshold, self.label,
                    self.left.copy() if self.left else None,
                    self.right.copy() if self.right else None,
                    self.fitness, self.id)

    def __str__(self):
        stack = [(self, 1)]
        output = ""

        while len(stack) > 0:
            node, depth = stack.pop()
            output += "-" * depth + " "

            if node.is_leaf():
                output += str(node.label)
            else:
                output += self.config['attributes'][node.attribute]
                output += " <= "
                output += '{:.5f}'.format(node.threshold)

                if node.right:
                    stack.append((node.right, depth + 1))
                if node.left:
                    stack.append((node.left, depth + 1))
            output += "\n"

        return output


if __name__ == "__main__":
    dataset = "breast_cancer"

    config = get_config(dataset)
    df = pd.read_csv(f"cro_dt/experiments/data/{dataset}.csv")
    feat_names = df.columns[:-1]
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=0, stratify=y_test)

    config["attributes_metadata"] = [(np.min(X_i), np.max(X_i)) for X_i in np.transpose(X_train.astype(np.float32))]
    tree = Tree.generate_random_tree(config, 2)
    print(tree)
    tree.mutate()
    print("\nMutated!\n")
    print(tree)
    tree.mutate()
    print("\nMutated!\n")
    print(tree)
    tree.mutate()
    print("\nMutated!\n")
    print(tree)
