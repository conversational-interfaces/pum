import json
import numpy as np

GT = '>'
LT = '<'


class Fluent:
    def __init__(self, type):
        self._type = type
        self._min = None
        self._max = None
        self._resolution = 5

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
        return self._resolution * (value - self._min) / (self._max - self._min)


class Model:
    def __init__(self, training_file):
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
        tensor = np.zeros(shape) + np.nan
        print(np.shape(tensor))
        print(tensor)
        for fluent_vec in self._favorable_fluent_vectors:
            tensor[self.fluent_vec_to_index(fluent_vec)] = 1
        for fluent_vec in self._unfavorable_fluent_vectors:
            tensor[self.fluent_vec_to_index(fluent_vec)] = 0
        print(tensor)

    def fluent_vec_to_index(self, fluent_vec):
        indices = []
        for name, value in sorted(fluent_vec.items()):
            index = int(self._fluent_dict[name].get_idx(value))
            print('{} -> {}'.format(value, index))
            indices.append(index)
        print(fluent_vec, indices)
        return indices

    def print_status(self):
        for name, fluent in self._fluent_dict.items():
            print(name, fluent)

    def train(self):
        pass

    def test(self, fluent_vector):
        return 0


if __name__ == '__main__':
    model = Model(training_file='pairs.json')
    model.print_status()
    model.train()
    utility = model.test({'numWrongAttempts': 0, 'success': True})
    print(utility)