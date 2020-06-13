

class FD(object):

    class Constant(object):
        def __init__(self, name, value):
            self.name = name
            self.value = value

    class InformationSource(object):
        pass

    class Expression(object):

        def __init__(self, elements_with_coefficients=None):
            self.elements = elements_with_coefficients if elements_with_coefficients != None else {}

        def __repr__(self):
            result = ''
            for key, value in self.elements.items():
                result += '{} : {}\n'.format(key, value)
            return result

        def __str__(self):
            result = ''
            for key, value in self.elements.items():
                result += '{} : {}\n'.format(key, value)
            return result

    class Flow(object):

        def __init__(self):
            self.start_point = 'None'
            self.end_point = 'None'

        def __repr__(self):
            result = 'flow : start : {},\tend : {}\n'.format(self.start_point, self.end_point)
            return result

        def __str__(self):
            result = 'flow : start : {},\tend : {}\n'.format(self.start_point, self.end_point)
            return result

    class Rate(object):

        def __init__(self, name):
            self.name = name
            self.expression = FD.Expression()
            self.flow = FD.Flow()
            self.names_units_map = None
            self.names_hidden_map = None

        def __repr__(self):
            result = 'rate : {}\n'.format(self.name)
            result += '{}\n'.format(str(self.flow))
            result += '{}\n'.format(str(self.expression))
            return result

        def __str__(self):
            result = 'name : {}\n'.format(self.name)
            result += '{}\n'.format(str(self.flow))
            result += '{}\n'.format(str(self.expression))
            return result

    def __init__(self, levels, constants, rates, dT):
        self.levels = levels
        self.constants = constants
        self.rates = rates
        self.dT = dT
        self.transform()

    def transform(self):
        units = [level for level in self.levels]
        units += [constant for constant in self.constants]

        hidden = [rate.name for rate in self.rates]

        names_units_map = {}
        names_hidden_map = {}

        for i, name in zip(range(len(units)), units):
            names_units_map[name] = i

        for i, name in zip(range(len(hidden)), hidden):
            names_hidden_map[name] = i

        self.names_units_map = names_units_map
        self.names_hidden_map = names_hidden_map
