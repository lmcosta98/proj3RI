# A class to store a heap node
class Node:
    def __init__(self, value, list_num, index):
        # `value` stores the element
        self.value = value
 
        # `list_num` stores the lists number of the element
        self.list_num = list_num
 
        # `index` stores the column number of the lists from which element was taken
        self.index = index
 
    # Override the `__lt__()` function to make `Node` class work with min-heap
    def __lt__(self, other):
        return self.value < other.value