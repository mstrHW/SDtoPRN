from pysd.py_backend.vensim.vensim2py import translate_vensim

from definitions import *
from module.fd_model.fd_model import FD

ONLY_OUT = 'only out'
IN_AND_OUT = 'in and out'
KNOWN_MODEL = 'known model'
UNKNOWN_MODEL = 'unknown model'

STOP_WORDS = ['INITIAL TIME', 'FINAL TIME', 'SAVEPER', 'Time', 'TIME']
TIME_STEP = 'TIME STEP'


def __parse_coefficients(elements_with_coefficients: Dict[str, int], expression: str, elements: List[str]) -> Dict[str, int]:
    for element in elements:
        position = expression.find(element)
        if position != -1:
            if '-' == expression[position - 2] or '-' == expression[position - 1]:
                coefficient = -1
            else:
                coefficient = 1
            elements_with_coefficients[element] = coefficient
    return elements_with_coefficients


def __get_fd_parameters(file_name: str) -> Dict:
    components = __get_components(file_name)
    levels, constants, rates, dt = __parse_components(components)
    components_with_expressions = __parse_expressions(components, levels, constants, rates)

    for level in levels:
        level_expression = components_with_expressions[level]
        for rate in rates:
            if rate.name in level_expression.elements.keys():
                if level_expression.elements[rate.name] < 0:
                    rate.flow.start_point = level
                else:
                    rate.flow.end_point = level
                rate.expression = components_with_expressions[rate.name]

    parameters =\
        {
            'levels': levels,
            'constants': constants,
            'rates': rates,
            'dt': dt
        }

    return parameters


def __get_components(file_name: str) -> List[Dict]:
    py_model_file, components = translate_vensim(file_name)
    return components


def __parse_components(components: List[Dict]) -> [List[str], List[str], List[FD.Rate], float]:
    levels = []
    constants = []
    rates = []
    dt = 0

    for component in components:
        name = component['real_name']
        kind = component['kind']
        if name in STOP_WORDS or kind == 'stateful':
            continue
        expression = component['expr']
        if name == TIME_STEP:
            dt = float(expression)
        elif kind == 'constant':
            constants.append(name)
            FD.Constant(name, float(expression))
        elif 'INTEG' in expression:
            levels.append(name)
        else:
            rates.append(FD.Rate(name))

    return levels, constants, rates, dt


def __parse_expressions(components: List[Dict], levels: List[str], constants: List[str], rates: List[FD.Rate]) -> Dict[str, FD.Expression]:
    components_with_expressions = {}

    for component in components:
        name = component['real_name']
        kind = component['kind']
        if name in STOP_WORDS or kind == 'stateful' or name == TIME_STEP:
            continue
        expression = component['expr']

        elements_with_coefficients = {}
        elements_with_coefficients = __parse_coefficients(elements_with_coefficients, expression, constants)
        elements_with_coefficients = __parse_coefficients(elements_with_coefficients, expression,
                                                           [rate.name for rate in rates])
        elements_with_coefficients = __parse_coefficients(elements_with_coefficients, expression, levels)
        my_expression = FD.Expression(elements_with_coefficients)
        components_with_expressions[name] = my_expression

    return components_with_expressions


def __generate_fd_parameters(names: List[str]) -> Dict:
    levels, constants, rates, dt = __generate_components(names)

    components_with_expressions = {}

    for level in levels:
        mode = ONLY_OUT
        # mode = IN_AND_OUT
        name1 = 'in_' + level
        name2 = 'out_' + level

        if mode == IN_AND_OUT:
            expression = name2 + '-' + name1
        else:
            expression = name1

        elements_with_coefficients = __parse_coefficients({}, expression, levels)
        components_with_expressions[level] = FD.Expression(elements_with_coefficients)
        components_with_coeff = dict([(_level, 1) for _level in levels])

        expr1 = FD.Expression(components_with_coeff)
        rate1 = FD.Rate(name1)
        rate1.flow.end_point = level
        rate1.expression = expr1
        rates.append(rate1)

        if mode == IN_AND_OUT:
            expr2 = FD.Expression(components_with_coeff)
            rate2 = FD.Rate(name2)
            rate2.flow.start_point = level
            rate2.expression = expr2
            rates.append(rate2)

    parameters =\
        {
            'levels': levels,
            'constants': constants,
            'rates': rates,
            'dt': dt
        }

    return parameters


def __generate_components(names: List[str]) -> [List[str], List, List, float]:
    levels = []
    constants = []
    rates = []
    dt = 0

    for name in names:
        if name not in STOP_WORDS:
            levels.append(name)

    return levels, constants, rates, dt


def get_fd(params, mode):
    if mode == KNOWN_MODEL:
        parameters = __get_fd_parameters(params)
    elif mode == UNKNOWN_MODEL:
        parameters = __generate_fd_parameters(params)
    else:
        print('unknown mode')
        return []
    levels = parameters['levels']
    constants = parameters['constants']
    rates = parameters['rates']
    dt = parameters['dt']
    fd_model = FD(levels, constants, rates, dt)
    return fd_model


def demo():
    models_directory = path_join(ROOT_DIR, 'vensim_models')
    file_name = path_join(models_directory, 'teacup.mdl')

    parameters = __get_fd_parameters(file_name)
    print(parameters)


if __name__ == '__main__':
    demo()
