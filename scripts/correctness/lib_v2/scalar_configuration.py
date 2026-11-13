"""Evaluate checked, compile-time API parameters from reference specializations.

The expression vocabulary is library-independent. Equations and acceptance
conditions live with the imported reference, never in an application recognizer.
This evaluates scalar metadata only; it does not establish a program match.
"""
import ast
import math
import operator
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


FUNCTIONS = {'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
             'floor': math.floor, 'ceil': math.ceil, 'abs': abs, 'f32': f32}
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
COMPARE = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def evaluate(expression, values):
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError('configuration expression must be bounded scalar text')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError('configuration expression is too complex')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.Name) and node.id in values:
            result = values[node.id]
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            result = BINARY[type(node.op)](visit(node.left), visit(node.right))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = visit(node.left), visit(node.right)
            if type(exponent) is not int or not 0 <= exponent <= 32 or abs(base) > 65536:
                raise ValueError('configuration power requires a bounded integer exponent')
            result = base ** exponent
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
            result = FUNCTIONS[node.func.id](visit(node.args[0]))
        elif isinstance(node, ast.Compare) and all(type(op) in COMPARE for op in node.ops):
            operands = [visit(node.left)] + [visit(v) for v in node.comparators]
            return all(COMPARE[type(op)](a, b)
                       for op, a, b in zip(node.ops, operands, operands[1:]))
        else:
            raise ValueError('unsupported scalar configuration expression')
        if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 2**63:
            raise ValueError('configuration value is nonfinite or out of range')
        return result

    try:
        return visit(tree.body)
    except (ArithmeticError, TypeError) as error:
        raise ValueError('invalid scalar configuration arithmetic') from error


def resolve(configuration, defines):
    if (not isinstance(configuration, dict)
            or set(configuration) != {'schema', 'inputs', 'values', 'checks'}
            or configuration['schema'] != 1):
        raise ValueError('unsupported scalar configuration schema')
    inputs, equations, checks = (configuration[k] for k in ('inputs', 'values', 'checks'))
    if (not isinstance(inputs, list) or len(inputs) != len(set(inputs))
            or not isinstance(equations, dict) or len(equations) > 128
            or not isinstance(checks, list) or not 1 <= len(checks) <= 128):
        raise ValueError('configuration requires inputs, ordered equations and checks')
    values = {}
    for name in inputs:
        if not isinstance(name, str) or name not in defines:
            raise ValueError('configuration input is not a reference specialization')
        values[name] = evaluate(str(defines[name]), {})
    for name, expression in equations.items():
        if (not isinstance(name, str) or not name.isidentifier()
                or name in values or name in FUNCTIONS):
            raise ValueError('invalid or duplicate configuration variable')
        value = evaluate(expression, values)
        if type(value) not in (int, float):
            raise ValueError('configuration variable must be numeric')
        values[name] = value
    for expression in checks:
        if evaluate(expression, values) is not True:
            raise ValueError('configuration check failed: ' + expression)
    return values


def literal(argument, values, element):
    if (not isinstance(argument, dict) or set(argument) != {'configuration', 'type'}
            or argument['type'] != element or element not in ('float', 'double', 'int')
            or argument['configuration'] not in values):
        raise ValueError('invalid configured wrapper argument')
    value = values[argument['configuration']]
    if element == 'int':
        if value != int(value) or not -2**31 <= value < 2**31:
            raise ValueError('configured argument does not fit int32')
        return str(int(value))
    if element == 'float' and f32(value) != value:
        raise ValueError('configuration must check the actual FP32 API value')
    return float(value).hex() + ('f' if element == 'float' else '')
