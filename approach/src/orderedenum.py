from enum import Enum


# https://docs.python.org/3/library/enum.html#orderedenum
class OrderedEnum(Enum):
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
