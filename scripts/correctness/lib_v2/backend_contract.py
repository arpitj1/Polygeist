"""Numerical backend compatibility, separate from source/reference equality.

Contracts describe a trusted library implementation model. Bounds below follow
from that model; hardware measurements do not prove the model for all inputs.
"""
from fractions import Fraction
import json
import math
import os


def budgets():
    try:
        result = json.loads(os.environ.get('POLYGEIST_BACKEND_ERROR_BUDGETS', '{}'))
    except (ValueError, TypeError) as error:
        raise ValueError('backend error budgets must be a JSON object') from error
    if (not isinstance(result, dict) or any(not isinstance(k, str)
            or type(v) not in (int, float) or not math.isfinite(v) or v < 0
            for k, v in result.items())):
        raise ValueError('backend error budgets require nonnegative finite values')
    return result


def approximations():
    """Explicitly accepted backend approximations with measured, unbounded error."""
    try:
        value = json.loads(os.environ.get('POLYGEIST_BACKEND_APPROXIMATIONS', '[]'))
    except (ValueError, TypeError) as error:
        raise ValueError('backend approximations must be a JSON list of specification names') from error
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ValueError('backend approximations must be a JSON list of specification names')
    return sorted(set(value))


def analyze(contract):
    if isinstance(contract, dict) and contract.get('kind') == 'recorded_approximation':
        if (set(contract) != {'schema', 'kind', 'source', 'assumptions'} or
                contract['schema'] != 1 or not isinstance(contract['source'], str) or
                not contract['source'].startswith('https://') or
                not isinstance(contract['assumptions'], list) or not contract['assumptions'] or
                any(not isinstance(x, str) or not x for x in contract['assumptions'])):
            raise ValueError('invalid recorded approximation contract')
        return dict(contract, basis='explicit backend approximation; no numerical error bound asserted')
    fields = {'schema', 'kind', 'source_weights', 'source_denominator',
              'backend_weights', 'backend_denominator', 'input_range',
              'source_rounding_error', 'backend_rounding_error', 'assumptions', 'source'}
    if (not isinstance(contract, dict) or set(contract) != fields or contract['schema'] != 1
            or contract['kind'] != 'bounded_linear_quantization'
            or not isinstance(contract['source'], str) or not contract['source'].startswith('https://')
            or not isinstance(contract['assumptions'], list) or not contract['assumptions']
            or any(not isinstance(a, str) or not a for a in contract['assumptions'])):
        raise ValueError('unsupported numerical backend contract')
    sw, bw = contract['source_weights'], contract['backend_weights']
    sd, bd = contract['source_denominator'], contract['backend_denominator']
    domain = contract['input_range']
    if (not isinstance(sw, list) or not isinstance(bw, list) or not 1 <= len(sw) <= 4096
            or len(sw) != len(bw) or any(type(v) is not int for v in sw + bw)
            or type(sd) is not int or type(bd) is not int or sd <= 0 or bd <= 0
            or not isinstance(domain, list) or len(domain) != 2
            or any(type(v) is not int for v in domain) or domain[0] > domain[1]):
        raise ValueError('invalid linear quantization coefficients or domain')
    # A difference in normalization is not treated as quantization noise.
    if sum(sw) != sd or sum(bw) != bd:
        raise ValueError('quantization contract must preserve normalization')
    low = high = Fraction(0)
    for s, b in zip(sw, bw):
        delta = Fraction(b, bd) - Fraction(s, sd)
        terms = [delta * x for x in domain]
        low += min(terms)
        high += max(terms)
    rounding = Fraction(0)
    for key in ('source_rounding_error', 'backend_rounding_error'):
        value = contract[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid rounding bound')
        rounding += Fraction(str(value))
    bound = max(abs(low), abs(high)) + rounding
    # Outputs are integers, so floor of the rational upper bound is safe.
    return dict(kind=contract['kind'], max_abs_error_bound=math.floor(bound),
                rational_error_bound=str(bound), input_range=domain,
                basis='interval bound conditional on the documented backend model',
                assumptions=contract['assumptions'], source=contract['source'])


def rejection(name, target, policy):
    contract = target.get('numerical_contract')
    if contract is None:
        return None
    try:
        info = analyze(contract)
    except ValueError as error:
        return str(error)
    if info['kind'] == 'recorded_approximation':
        return None if name in approximations() else 'backend approximation requires explicit acceptance'
    if name not in policy:
        return 'approximate backend requires an explicit error budget'
    if policy[name] < info['max_abs_error_bound']:
        return 'backend error budget is smaller than the modeled error bound'
    return None


def evidence(name, target, policy):
    if 'numerical_contract' not in target:
        return None
    reason = rejection(name, target, policy)
    if reason:
        raise ValueError(reason)
    info = analyze(target['numerical_contract'])
    acceptance = (dict(accepted_approximation=True) if info['kind'] == 'recorded_approximation'
                  else dict(accepted_error_budget=policy[name]))
    return dict(info, **acceptance,
                source_reference_match='exact under the selected scalar arithmetic model',
                vendor_execution='requires independent validation')
