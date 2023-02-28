from typing import List


class UnionFind:
    def __init__(self, data: List, union_condition: callable):
        self.__data__ = data
        self.__union_condition__ = union_condition
        length = len(data)
        self.__parents__ = [i for i in range(length)]
        self.__ranks__ = [0] * length
        self.__unions__ = {}

    def __find_parent__(self, id: int):
        return self.__parents__[id]

    def __find_root__(self, id: int):
        parent = self.__find_parent__(id)
        while parent != id:
            id = parent
            parent = self.__find_parent__(id)
        return id

    def __union__(self, i: int, j: int):
        root_i = self.__find_root__(i)
        root_j = self.__find_root__(j)

        # if roots are different, let one be the parent of the other
        if root_i == root_j:
            return
        else:
            if self.__ranks__[root_i] <= self.__ranks__[root_j]:
                # root of i --> child
                self.__parents__[root_i] = root_j
                self.__ranks__[root_j] = max(self.__ranks__[root_j], self.__ranks__[root_i]+1)
            else:
                self.__parents__[root_j] = root_i
                self.__ranks__[root_i] = max(self.__ranks__[root_i], self.__ranks__[root_j]+1)

    def union_step(self):
        length = len(self.__data__)

        for i in range(length - 1):
            for j in range(i + 1, length):
                if self.__union_condition__(self.__data__[i], self.__data__[j]):
                    self.__union__(i, j)

        for i in range(length):
            root = self.__find_root__(i)
            if root not in self.__unions__.keys():
                self.__unions__[root] = [self.__data__[i]]
            else:
                self.__unions__[root].append(self.__data__[i])

    def get_unions(self):
        return self.__unions__
