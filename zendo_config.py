from collections import defaultdict
from type_system import Arrow, List, INT, BOOL
from program import Function, Variable, BasicPrimitive
from DSL.zendo import primitive_types
from DSL.zendo import STRUCTURE
from types import SimpleNamespace

def flatten_input_types(arrow):
    types = []
    while isinstance(arrow, Arrow):
        types.append(arrow.input_type)
        arrow = arrow.output_type
    return types

cfg = SimpleNamespace()
cfg.start = "BOOL"
cfg.max_program_depth = 4  # or whatever depth you want
cfg.rules = defaultdict(dict)

# Add terminals (constants, primitives)
for name, type_ in primitive_types.items():
    if isinstance(type_, Arrow):
        output_type = type_.returns()     # get return type
        input_types = type_.arguments()   # get input type list
        cfg.rules[output_type][name] = input_types
    else:
        cfg.rules[type_][name] = []

# Add example: compose two BOOL functions using AND or OR
cfg.rules[BOOL]["AND"] = [BOOL, BOOL]
cfg.rules[BOOL]["OR"] = [BOOL, BOOL]
cfg.rules[BOOL]["EITHER_OR"] = [INT, INT, List(STRUCTURE)]
cfg.use_rules = True
