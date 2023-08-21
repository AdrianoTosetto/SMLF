import json
from functools import reduce
from itertools import starmap


def json_style_to_css(json: dict) -> str:

    kv_to_string = lambda key, value: '{key}: {value}'.format(key=key, value=value)
    style_props = list(starmap(kv_to_string, zip(json.keys(), json.values())))

    return reduce(lambda acc, curr: '{acc}{curr};'.format(acc=acc, curr=curr), style_props, '')

def json_style_to_id_css(id: str,json: dict):
    return '#{id} {{ {style} }}'.format(id=id, style=json_style_to_css(json))
