def field(items, *args):
    assert len(args) > 0
    if len(args) == 1:
        for d in items:
            if args[0] in d and d[args[0]] != None:
                yield d[args[0]]
    else:
        for d in items:
            d_item = dict()
            for key in args:
                if key in d and d[key] != None:
                    d_item[key] = d[key]
            if len(d_item) > 0:
                yield d_item
    
goods = [
{'title': 'Ковер', 'price': 2000, 'color': 'green'},
{'title': 'Диван для отдыха', 'price': 5300, 'color': 'black'},
{'title': 'None', 'price': None, 'color': 'yellow'}
]
print(*field(goods, 'title'))
print(*field(goods, 'title', 'price'))