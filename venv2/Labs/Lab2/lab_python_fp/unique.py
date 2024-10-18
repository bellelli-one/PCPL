from gen_random import gen_random
import types

class generatortoiter(object):
    def __init__(self, gen):
        self.gen = gen
    
    def __iter__(self):
        return self.gen()


class Unique(object):
    def __init__(self, items, **kwargs):
        if isinstance(items, types.GeneratorType):
            self.data = iter(*items)
        else :
            self.data = iter(items)
        self.ignore_case = kwargs.get('ignore_case', False)
        self.unique_items = set()

    def __next__(self):
        while True:
            item = next(self.data)
            check_item = item.lower() if self.ignore_case else item
            if check_item not in self.unique_items:
                self.unique_items.add(check_item)
                return item
            
    def __iter__(self):
        return self
    
data = gen_random(10, 1, 3)
unique_data = Unique(data)
print(list(unique_data))