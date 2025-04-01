from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable
import numpy as np
from scipy.special import logsumexp


@dataclass
class EProcess:
    """
    A class representing a stochastic process with a time series of values

    Attributes:
    _values: np.ndarray
        The time series of values

    Methods:
    __getitem__(item: int) -> float:
        Returns the value at the given index
    __len__() -> int:
        Returns the length of the time series
    append(values: np.ndarray) -> None:
        Appends the given values to the end of the time series
    append_at(values: np.ndarray, start: int) -> None:
        Appends the given values to the time series starting at the given index.
        Requires that the number of values to append is at least as long as the time series minus the start index
    """
    _values: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    # TODO: log scale or natural scale?
    # TODO: constructor check that first value is 1 (natrual scale) or 0 (log scale)

    def __getitem__(self, item):
        try:
            return self._values[item]
        except IndexError:
            raise IndexError(f"Index {item} out of bounds for EProcess with shape {self._values.shape}")

    def __len__(self):
        return len(self._values)

    def append(self, values: np.ndarray):
        # TODO check non-negative
        self._values = np.append(self._values, values)

    def append_at(self, values: np.ndarray, start: int):
        # TODO check non-negative
        assert start <= len(self._values), f"Index {start} out of bounds for EProcess with shape {self._values.shape}"
        assert len(values) + start >= len(self._values), f"Not enough values to append at index {start}"
        self._values[start:] = values[:len(self._values) - start]
        self.append(values[len(self._values) - start:])

    def pvalue(self):
        pass


class Combiner(ABC):
    """
    A class for combiner a series of stochastic processes into a single process

    Attributes:
    history: np.ndarray
        The history of values of the composed process
    history_length: int
        The length of the history to keep

    Methods:
    __call__(values: np.ndarray) -> float:
        Objects of this class are callable. Given an array of values, it composes them into a single value
    history_append(values: np.ndarray) -> None:
        Appends the given values to the history
    set_history(values: np.ndarray) -> None:
        Sets the history to the given values
    operate(values: np.ndarray, increments: np.ndarray) -> float:
        Abstract method that must be implemented by subclasses. Given the values and the increments, it returns a
        single value
    """
    def __init__(self, history_length: int = 1):
        self.history = None
        if history_length < 0:
            raise ValueError("history_length must be non-negative")
        self.history_length = history_length  # non-negative

    def __call__(self, values: np.ndarray) -> float:
        """
        Composes the 2D array of values into an array of values of the same length
        """
        if self.history is None:
            self.history_append(values)
            return 0
        else:
            increments = values - self.history[-1]
            combination = self.operate(values, increments)
            self.history_append(values)
            return combination

    def history_append(self, values: np.ndarray):
        if self.history is None:
            self.history = np.array([values])
        else:
            self.history = np.append(self.history, [values], axis=0)
            if len(self.history) > self.history_length:
                self.history = self.history[-self.history_length:]

    def set_history(self, values: np.ndarray):
        self.history = values[:self.history_length]

    def get_start_with_history(self, start: int) -> int:
        return max(0, start - self.history_length)

    @abstractmethod
    def operate(self, values: np.ndarray, increments: np.ndarray) -> float:
        pass


class Minimum(Combiner):
    def operate(self, values: np.ndarray, increments: np.ndarray) -> float:
        # values[i] is [eproc1[i], eproc2[i], ...]
        return np.min(values, axis=0)


class Linear(Combiner):
    def operate(self, values: np.ndarray, increments: np.ndarray) -> float:
        # print(self.history[-1], increments)
        # print(increments + self.history[-1])
        # print(logsumexp(increments + self.history[-1]))
        # print(logsumexp(self.history[-1]))
        # print(logsumexp(increments + self.history[-1]) - logsumexp(self.history[-1]))
        # print(f" --- {self.history[-1]} + {increments} = {increments + self.history[-1]}")
        return logsumexp(increments + self.history[-1]) - logsumexp(self.history[-1])


class NodeStatus(Enum):
    ACTIVE = auto(),
    PARKED = auto(),
    ABANDONED = auto(),
    REJECTED = auto(),
    # PRUNED = auto()


@dataclass
class Node:
    id: str
    eprocess: EProcess = field(default_factory=lambda: EProcess())
    children: list['Node'] = None
    parents: list['Node'] = None
    status: NodeStatus = NodeStatus.ACTIVE
    combiner: Combiner = None

    def __getitem__(self, item):
        if item >= len(self.eprocess):
            self.combine(start=len(self.eprocess), end=item+1)
        return self.eprocess[item]

    def combine(self, start: int = 0, end: int = -1) -> None:
        """
        Combines the values of children processes into a single process for indices start <= i < end
        """
        hist_start = self.combiner.get_start_with_history(start)
        hist_len = start - hist_start

        # collate the values of the children processes
        transposed = np.stack([child.eprocess[hist_start:end] for child in self.children], axis=1)

        # set the history of the combiner
        self.combiner.set_history(transposed[:hist_len])

        # remove the history
        transposed = transposed[self.combiner.history_length:]

        # combine the values and bets
        increments = np.array(list(map(self.combiner, transposed)))

        # append the values to the process
        values = np.cumsum(increments) + self.eprocess[start-1]
        self.eprocess.append_at(values, start)


@dataclass
class NodeRegistry:
    nodes: dict[str, Node] = field(default_factory=lambda: {})

    def __getitem__(self, item):
        if item in self.nodes:
            return self.nodes[item]
        else:
            self.nodes[item] = Node(item)
            return self.nodes[item]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    # n_processes: int = 0
    # n_forgotten: int = 0
    # children: list['EProcess'] = None
    # operator: 'EProcessOperation' = None


# EProcessOperation = Callable[[list[EProcess]], EProcess]
#
#
# def weighted_average_linear(eprocesses: list[EProcess]) -> EProcess:
#     values = np.zeros(eprocesses[0].values.shape)
#     for eprocess in eprocesses:
#         values += eprocess.values
#     values /= len(eprocesses)
#     return EProcess(values)
#
#


if __name__ == '__main__':
    eproc1 = EProcess(np.array([0, -1.75073322065061, -2.32966210331461, -2.4687480351667, -2.08241577969098,
                                -1.56554184160148, -1.04753438222735, -1.22482110067968, -2.33378315213113,
                                -1.91927664645909, -2.0279242996295, -2.08021631576895, -3.92701680958911,
                                -4.00145656165818, -4.11331171409417, -3.49762445409762, -3.42694405795045,
                                -3.11781275247875, -3.1314236493364, -1.62419933233654, -3.23536677319802,
                                -2.63119630314807, -2.86186749654706, -3.52196118900269, -1.85854237049017,
                                -0.785966835722843]))
    eproc2 = EProcess(np.array([0, 2.06180902713567, 3.90192375764514, 5.81307764973515, 6.46144242314048,
                                5.98314710616874, 6.16543847618669, 7.59368074919381, 7.55547550346757,
                                9.34246624558819, 8.9148204982283, 11.3428002569942, 12.1167138853506,
                                15.8441964422166, 16.8768171496276, 18.4874713644601, 20.8472594267053,
                                21.2271332861065, 23.3036481521497, 25.4520719131317, 26.1870639950041,
                                27.3600058062842, 31.8526307855986, 33.069414390422, 32.7292703086017,
                                32.927677305119]))
    eproc3 = EProcess(np.array([0, 0.617547479396937, -2.42879677963893, -3.0960536915673, -2.6987024876918,
                                -2.75512514819675, -3.51107258098083, -3.67847432049237, -2.90954692762666,
                                -1.12050346391258, -3.5665321191454, -2.81343129301966, -2.97349983445838,
                                -3.27521174863272, -4.04658837085074, -3.27246643504547, -4.08059594617727,
                                -4.10714187644974, -2.15558948791704, -3.47582956195296, -2.86011340121876,
                                -3.11847745200669, -5.93535836051045, -2.93800365176913, -3.06928988326519,
                                -2.64145675293726]))
    eproc4 = EProcess(np.array([0, -0.737673551292295, -1.27460356454169, -2.61663123497134, -2.68994561270798,
                                -2.77164306444692, -3.50103072405301, -3.62981272095649, -4.59255604079049,
                                -5.79431375165014, -7.43105828191427, -8.1124271975788, -8.54539041066458,
                                -10.2156547817377, -10.6102484164414, -11.4170580551689, -12.2290583616552,
                                -12.0034328617269, -12.2450981469484, -12.3612197030959, -12.1969950926771,
                                -12.1054540483381, -11.722309264415, -12.3603026212464, -12.9200823110778,
                                -12.8562454861131]))
    eproc5 = EProcess(np.array([0, 1.58339906447881, 2.27594140565772, 2.96899767891929, 3.18070393712666,
                                3.43046022846246, 3.4249992077154, 3.91409534761015, 4.9469007658416,
                                4.7064222641394, 5.32406389725611, 5.85694263399254, 6.27556455548034,
                                7.07448001047541, 8.51285946915221, 9.23247376707505, 9.36431857827925,
                                10.5955858920775, 11.2510070091401, 12.846611503395, 12.854244922352,
                                13.4628123711735, 13.0952524984021, 13.9812479385424, 14.2584566889486,
                                15.0316984784867]))
    reg = NodeRegistry()
    reg['eproc1'] = Node('eproc1', eproc1)
    reg['eproc2'] = Node('eproc2', eproc2)
    reg['eproc3'] = Node('eproc3', eproc3)
    reg['eproc4'] = Node('eproc4', eproc4)
    reg['eproc5'] = Node('eproc5', eproc5)
    reg['parent'] = Node('parent', combiner=Minimum())
    reg['parent'].children = [reg['eproc1'], reg['eproc2'], reg['eproc3'], reg['eproc4'], reg['eproc5']]
    # print(reg['parent'][10])
    print("3:", reg['parent'][3])
    # print("mid:", reg['parent'][7])
    print("10:", reg['parent'][10])
    #print(reg['parent'][9], reg['parent'].eprocess._values)

    # x = compose([eproc1, eproc2, eproc3], start=0)
    # print(x)
    pass
    # todo add tests


# [0.         0.         1.11434841 2.47988864 4.26077053 4.26077053 3.82094559 3.99081967 3.99081967 3.99865728 5.72420566]
# [0.         0.         1.11434841 2.47988864 4.26077053 4.26077053 3.82094559 3.99081967 5.38079695 5.38863457 7.11418294]