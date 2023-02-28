from typing import List, Iterable, Union
from pprint import pprint

class KeyphraseCount:

    def __init__(self, keyphrase: str, count: int) -> None:
        super().__init__()
        self.keyphrase = keyphrase
        self.count = count

    @classmethod
    def reduce(cls, kcs: list) :
        '''
        kcs: List[KeyphraseCount]
        '''
        keys = ''
        count = 0

        for i in range(len(kcs)-1):
            kc = kcs[i]
            keys += kc.keyphrase + '/'
            count += kc.count

        keys += kcs[-1].keyphrase
        count += kcs[-1].count
        return KeyphraseCount(keys, count)



class SingleCluster:
    def __init__(self):
        self.__container__ = []
        self.__keyphrases__ = {}
    def add(self, id:int):
        self.__container__.append(id)
    def __str__(self) -> str:
        return str(self.__container__)
    def elements(self) -> List:
        return self.__container__
    def get_keyphrases(self):
        ret = []
        for key, count in self.__keyphrases__.items():
            ret.append(KeyphraseCount(key,count))
        return ret
    def add_keyphrase(self, keyphrase:Union[str,Iterable]):
        if isinstance(keyphrase,str):
            if keyphrase not in self.__keyphrases__.keys():
                self.__keyphrases__[keyphrase] = 1
            else:
                self.__keyphrases__[keyphrase] += 1
        elif isinstance(keyphrase,Iterable):
            for i in keyphrase:
                self.add_keyphrase(i)

    def __len__(self):
        return len(self.__container__)

    def print_keyphrases(self):
        pprint(self.__keyphrases__)

class ClusterList:
    def __init__(self, k:int):
        self.__clusters__ = [SingleCluster() for _ in range(k)]

    # subscriptable and slice-able
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.__clusters__[idx]
        if isinstance(idx, slice):
            # return
            return self.__clusters__[0 if idx.start is None else idx.start: idx.stop: 0 if idx.step is None else idx.step]

    def instantiate(self, labels: Iterable):
        for id, label in enumerate(labels):
            self.__clusters__[label].add(id)

    def __str__(self):
        ret = f'There are {len(self.__clusters__)} clusters:\n'
        for id,cluster in enumerate(self.__clusters__):
            ret += f'cluster {id} contains: {cluster}.\n'

        return ret

    # return an iterator that can be used in for loop etc.
    def __iter__(self):
        return self.__clusters__.__iter__()

    def __len__(self): return len(self.__clusters__)

    def sort(self):
        self.__clusters__.sort(key=len,reverse=True)
