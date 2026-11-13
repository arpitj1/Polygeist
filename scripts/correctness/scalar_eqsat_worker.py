"""Isolated, bounded distributive scalar proof; input/output are JSON on stdio."""
import json
import resource
import sys
import time

import kernel_match as K
import eqsat_ablation as A


def main():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    request = json.load(sys.stdin)
    if K.numeric_model() != 'relaxed':
        raise ValueError('Distributive scalar proofs require relaxed arithmetic')

    def tree(value):
        return tuple(map(tree, value)) if isinstance(value, list) else value

    left, right = [K._ast_to_term(tree(request[key])) for key in ('left', 'right')]
    graph = K.EGraph()
    graph.register(left, right)
    # Subtraction is real addition with an additive inverse. Keep this rule
    # confined to the isolated relaxed proof, alongside distributivity.
    mode, excluded = A.mode()
    if mode == 'none':
        print(json.dumps(dict(matched=False, iterations=0)))
        return
    rules = K.algebra_rules(False) if mode == 'full' else A.scalar_ruleset(K, excluded)
    if 'distrib' not in excluded:
        rules = rules | K.ruleset(
            K.rewrite(K.a * (K.b + K.c)).to(K.a * K.b + K.a * K.c),
            K.rewrite((K.a + K.b) * K.c).to(K.a * K.c + K.b * K.c),
            K.rewrite(K.a - K.b).to(K.a + K.Term.Lit(-1.0) * K.b))
    matched = False
    rule_matches = 0
    started = time.perf_counter()
    for iteration in range(8):
        report = graph.run(rules * 1)
        rule_matches += sum(report.num_matches_per_rule.values())
        try:
            graph.check(left == right)
            matched = True
            break
        except Exception:
            pass
    result = dict(matched=matched, iterations=iteration + 1,
                  saturation_elapsed_ms=(time.perf_counter()-started)*1000,
                  rule_matches=rule_matches)
    if request.get('collect_sizes'):
        serialized = json.loads(graph._serialize(
            split_primitive_outputs=False, include_temporary_functions=True).to_json())
        result['nodes'] = len(serialized.get('nodes', {}))
        result['classes'] = len(serialized.get('class_data', {}))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
