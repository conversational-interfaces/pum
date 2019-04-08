import json
import numpy as np
from sklearn import linear_model

GT = '>'
LT = '<'


class Fluent:
    def __init__(self, type):
        self._type = type
        self._min = None
        self._max = None
        self._resolution = 6

    def __str__(self):
        return '[{}, {}]'.format(self._min, self._max)

    def update(self, val):
        if self._min is None:
            self._min = val
            self._max = val
        if val < self._min:
            self._min = val
        if val > self._max:
            self._max = val

    def range(self):
        if self._type is bool:
            return [0, 1]
        else:
            return list(range(self._resolution))

    def get_idx(self, value):
        if self._type is bool:
            if value:
                return True
            else:
                return False
        #   self.fluent_vec_to_index(fluent_vec)
        return (self._resolution - 1) * (value - self._min) / (self._max - self._min)

    def get_value(self, index):
        if self._type is bool:
            if index:
                return True
            else:
                return False
        return self._min + (self._max - self._min) / (self._resolution - 1) * index

class Model:
    def __init__(self, training_file):
        self._regression_model = None
        self._fluent_dict = {}
        self._favorable_fluent_vectors = []
        self._unfavorable_fluent_vectors = []
        pairs = json.load(open(training_file))
        for op, f1, f2 in pairs:
            if op == GT:
                self._favorable_fluent_vectors.append(f1)
                self._unfavorable_fluent_vectors.append(f2)
            elif op == LT:
                self._unfavorable_fluent_vectors.append(f1)
                self._favorable_fluent_vectors.append(f2)

            for name, val in list(f1.items()):
                if name not in self._fluent_dict:
                    self._fluent_dict[name] = Fluent(type(val))
                self._fluent_dict[name].update(val)
            for name, val in list(f2.items()):
                if name not in self._fluent_dict:
                    self._fluent_dict[name] = Fluent(type(val))
                self._fluent_dict[name].update(val)

        shape = []
        for name, fluent in sorted(self._fluent_dict.items()):
            shape.append(len(fluent.range()))
        self.input_tensor = np.zeros(shape) + np.nan
        # print(np.shape(tensor))
        print(self.input_tensor)
        print("prefer fluents")
        for fluent_vec in self._favorable_fluent_vectors:
            self.input_tensor[self.fluent_vec_to_index(fluent_vec)] = 1
        print("avoid fluents")
        for fluent_vec in self._unfavorable_fluent_vectors:
            self.input_tensor[self.fluent_vec_to_index(fluent_vec)] = 0
        print(self.input_tensor)

    def fluent_vec_to_index(self, fluent_vec):
        indices = []
        for name, value in sorted(fluent_vec.items()):
            index = int(self._fluent_dict[name].get_idx(value))
            # print('{} -> {}'.format(value, index))
            indices.append(index)
        print(fluent_vec, indices)
        return tuple(indices)

    def fluent_vec_to_np_vec(self, fluent_vec):
        vec = []
        for name, value in sorted(fluent_vec.items()):
            vec.append(int(value))
        return np.array(vec)

    def index_to_np_vec(self, indices):
        values = []
        sorted_fluents = sorted(self._fluent_dict.items())
        for i, index in enumerate(indices):
            values.append(sorted_fluents[i][1].get_value(index))
        return np.array(values)

    def print_status(self):
        for name, fluent in self._fluent_dict.items():
            print(name, fluent)

    def train(self):
        print("start training")
        output_tensor = self.floodfill(self.input_tensor)
        # output_tensor = self.input_tensor
        X = []
        y = []
        for index in np.ndindex(output_tensor.shape):
            val = output_tensor[index]
            if not np.isnan(val):
                X.append(np.array(index))
                y.append(val)

        self._regression_model = linear_model.LinearRegression()
        self._regression_model.fit(X, y)

        print(self._regression_model.coef_)
        print(self._regression_model.intercept_)

        print("-----")

    def test(self, fluent_vector):
        print(fluent_vector)
        x = self.fluent_vec_to_np_vec(fluent_vector)
        return self._regression_model.predict(np.reshape(x, (1, -1)))

    def test2(self):
        ret = np.zeros_like(self.input_tensor)
        for index in np.ndindex(self.input_tensor.shape):
            x = self.index_to_np_vec(index)
            ret[index] = self._regression_model.predict(np.reshape(x, (1, -1)))
        print(ret)

    def floodfill(self, input_tensor):  # input tensor is not modified
        iters = 10
        output_tensor = np.copy(input_tensor)
        shape = output_tensor.shape

        for i in range(iters):
            for index in np.ndindex(shape):
                if np.isnan(self.input_tensor[index]):  # don't modify 0 and 1 in the original input tensor
                    neighbors = []  # array of index, each index a tuple, 1 connectivity
                    for j in range(len(index)):
                        temp1 = index[:j] + (index[j]+1, ) + index[j+1:]
                        if 0 <= temp1[j] < shape[j]:
                            neighbors.append(temp1)
                        temp2 = index[:j] + (index[j]-1, ) + index[j+1:]
                        if 0 <= temp2[j] < shape[j]:
                            neighbors.append(temp2)
                    new_val = 0
                    count = 0
                    # average of (neighbor * 0.9), excluding nan neighbors
                    for neighbor in neighbors:
                        temp = output_tensor[neighbor]
                        if not np.isnan(temp):
                            count += 1
                            new_val += 0.9 * temp
                    if count:
                        output_tensor[index] = new_val / count
                    else:
                        output_tensor[index] = np.nan

            print("iteration ", i)
            print(output_tensor)
        return output_tensor



if __name__ == '__main__':
    model = Model(training_file='pairs.json')
    model.print_status()
    model.train()
    utility = model.test({'numWrongAttempts': 0, 'success': True})
    print(utility)
    model.test2()

