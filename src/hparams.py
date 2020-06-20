import json


class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def repr__(self):
        return '{%s}' % str('\n '.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

    def add_hparam(self, name, value):
        self.__dict__.update({name: value})

    def del_hparam(self, name):
        if name in self.__dict__:
            self.__dict__.pop(name)
        else:
            print("There is no parameter named %s" % name)

    def get(self, key, default=None):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return default

    def override_from_dict(self, values_dict):
        if not isinstance(values_dict, dict):
            raise TypeError("Excepting argument with type 'dict', but receive %s" % type(values_dict))
        self.__dict__.clear()
        self.__dict__.update(values_dict)

    @staticmethod
    def _convert_to_num(value):
        new_value = None
        try:
            if value == 'NaN':
                flag = False
            else:
                new_value = float(value)
                flag = '.' in value
        except ValueError:
            flag = False
        if not flag:
            if value.isdigit():
                new_value = int(value)
                flag = True
            else:
                flag = False
        if flag:
            return new_value
        else:
            return value

    def parse(self, values):
        if type(values) is not str:
            raise TypeError("Excepting argument with type 'str', but receive %s" % type(values))
        items = values.split(',')
        values_dict = dict()
        for item in items:
            (key, value) = item.split('=')
            values_dict[key] = self._convert_to_num(value)
        self.override_from_dict(values_dict)
        return self

    def parse_json(self, values_json):
        if type(values_json) is not str:
            raise TypeError("Excepting argument with type 'str', but receive %s" % type(values_json))
        values_dict = json.loads(values_json)
        self.override_from_dict(values_dict)
        return self

    def set_hparam(self, name, value):
        if name in self.__dict__:
            tn = type(value)
            to = type(self.__dict__[name])
            if tn == to:
                self.__dict__[name] = value
            else:
                raise TypeError("Excepting value with type %s, but receive %s" % (to, tn))
        else:
            print("There is no parameter named %s" % name)

    def to_json(self, indent=None, separators=None, sort_keys=False):
        return json.dumps(self.__dict__, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    def values(self):
        return self.__dict__