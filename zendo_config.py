from collections import defaultdict
from type_system import Arrow, List, INT, BOOL, PrimitiveType
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
cfg.start = PrimitiveType("bool")
cfg.max_program_depth = 4  # or whatever depth you want
cfg.rules = defaultdict(dict)

# Add terminals (constants, primitives)
for name, type_ in primitive_types.items():
    if isinstance(type_, Arrow):
        output_type = type_.returns()     # get return type
        input_types = type_.arguments()   # get input type list
        cfg.rules[output_type][BasicPrimitive(name)] = input_types
    else:
        cfg.rules[type_][BasicPrimitive(name)] = []

cfg.rules[BOOL][Function("AND", Arrow(BOOL, Arrow(BOOL, BOOL)))] = [BOOL, BOOL]
cfg.rules[BOOL][Function("OR", Arrow(BOOL, Arrow(BOOL, BOOL)))] = [BOOL, BOOL]
cfg.rules[BOOL][Function("EITHER_OR", Arrow(INT, Arrow(INT, Arrow(List(STRUCTURE), BOOL))))] = [INT, INT, List(STRUCTURE)]

# Boolean constants
cfg.rules[BOOL][BasicPrimitive(True)] = []
cfg.rules[BOOL][BasicPrimitive(False)] = []

# Integer constants
cfg.rules[INT][BasicPrimitive(0)] = []
cfg.rules[INT][BasicPrimitive(1)] = []
cfg.rules[INT][BasicPrimitive(2)] = []

# Dummy structure and structure lists
cfg.rules[STRUCTURE][BasicPrimitive("x")] = []
cfg.rules[List(STRUCTURE)][BasicPrimitive(["x", "y"])] = []

# Predicates
for name in ["IS_RED", "IS_BLUE", "IS_YELLOW"]:
    cfg.rules[Arrow(STRUCTURE, BOOL)][Function(name, Arrow(STRUCTURE, BOOL))] = [STRUCTURE]

for name in ["IS_BLOCK", "IS_PYRAMID", "IS_WEDGE"]:
    cfg.rules[Arrow(STRUCTURE, BOOL)][Function(name, Arrow(STRUCTURE, BOOL))] = [STRUCTURE]

for name in ["IS_UPRIGHT", "IS_FLAT", "IS_UPSIDE_DOWN", "IS_CHEESECAKE"]:
    cfg.rules[Arrow(STRUCTURE, BOOL)][Function(name, Arrow(STRUCTURE, BOOL))] = [STRUCTURE]

# Count
cfg.rules[BOOL][Function("COUNT", Arrow(Arrow(STRUCTURE, BOOL), Arrow(List(STRUCTURE), INT)))] = [Arrow(STRUCTURE, BOOL), List(STRUCTURE)]

cfg.use_rules = True