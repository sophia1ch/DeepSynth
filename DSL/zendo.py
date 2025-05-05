from type_system import *
from program import *

# Define concrete type for Zendo pieces
STRUCTURE = PrimitiveType("structure")
t0 = STRUCTURE

# ---- Semantic functions ----
def count_predicate(pred, structure):
    return len([piece for piece in structure if pred(piece, structure)])

def has_color(color):
    return lambda piece, _: piece['color'] == color

def has_shape(shape):
    return lambda piece, _: piece['shape'] == shape

def has_orientation(orientation):
    return lambda piece, _: piece['orientation'] == orientation

def touching(pred):
    def pred_fn(piece, structure):
        for direction, other_id in piece.get('touching', {}).items():
            other_piece = next((p for p in structure if p['ID'] == other_id), None)
            if other_piece and pred(other_piece, structure):
                return True
        return False
    return pred_fn

def pointing(pred):
    def pred_fn(piece, structure):
        pointed_id = piece.get('pointing', "")
        if not pointed_id:
            return False
        other_piece = next((p for p in structure if p['ID'] == pointed_id), None)
        return other_piece and pred(other_piece, structure)
    return pred_fn

def and_rule(rule1, rule2):
    return lambda structure: rule1(structure) and rule2(structure)

def or_rule(rule1, rule2):
    return lambda structure: rule1(structure) or rule2(structure)

def either_or(n1, n2, structure):
    return len(structure) == n1 or len(structure) == n2

# ---- DSL Semantics ----
semantics = {
    'COUNT': lambda pred: lambda structure: count_predicate(pred, structure),
    'AT_LEAST': lambda n: lambda pred: lambda structure: count_predicate(pred, structure) >= n,
    'EXACTLY': lambda n: lambda pred: lambda structure: count_predicate(pred, structure) == n,
    'EVEN': lambda pred: lambda structure: count_predicate(pred, structure) % 2 == 0,
    'ODD': lambda pred: lambda structure: count_predicate(pred, structure) % 2 == 1,
    'MORE_THAN': lambda pred1: lambda pred2: lambda structure:
        count_predicate(pred1, structure) > count_predicate(pred2, structure),
    'EITHER_OR': either_or,
    'AND': and_rule,
    'OR': or_rule,

    # Basic unary predicates
    'IS_RED': has_color('red'),
    'IS_BLUE': has_color('blue'),
    'IS_YELLOW': has_color('yellow'),
    'IS_BLOCK': has_shape('block'),
    'IS_PYRAMID': has_shape('pyramid'),
    'IS_WEDGE': has_shape('wedge'),
    'IS_UPRIGHT': has_orientation('upright'),
    'IS_FLAT': has_orientation('flat'),
    'IS_UPSIDE_DOWN': has_orientation('upside_down'),
    'IS_CHEESECAKE': has_orientation('cheesecake'),

    # Interactions
    'TOUCHING': touching,
    'POINTING': pointing,
}

# ---- DSL Type Signatures ----
primitive_types = {
    'COUNT': Arrow(Arrow(t0, BOOL), Arrow(List(t0), INT)),
    'AT_LEAST': Arrow(INT, Arrow(Arrow(t0, BOOL), Arrow(List(t0), BOOL))),
    'EXACTLY': Arrow(INT, Arrow(Arrow(t0, BOOL), Arrow(List(t0), BOOL))),
    'EVEN': Arrow(Arrow(t0, BOOL), Arrow(List(t0), BOOL)),
    'ODD': Arrow(Arrow(t0, BOOL), Arrow(List(t0), BOOL)),
    'MORE_THAN': Arrow(Arrow(t0, BOOL), Arrow(Arrow(t0, BOOL), Arrow(List(t0), BOOL))),
    'EITHER_OR': Arrow(INT, Arrow(INT, Arrow(List(t0), BOOL))),
    'AND': Arrow(Arrow(List(t0), BOOL), Arrow(Arrow(List(t0), BOOL), Arrow(List(t0), BOOL))),
    'OR': Arrow(Arrow(List(t0), BOOL), Arrow(Arrow(List(t0), BOOL), Arrow(List(t0), BOOL))),

    # Unary predicates
    'IS_RED': Arrow(t0, BOOL),
    'IS_BLUE': Arrow(t0, BOOL),
    'IS_YELLOW': Arrow(t0, BOOL),
    'IS_BLOCK': Arrow(t0, BOOL),
    'IS_PYRAMID': Arrow(t0, BOOL),
    'IS_WEDGE': Arrow(t0, BOOL),
    'IS_UPRIGHT': Arrow(t0, BOOL),
    'IS_FLAT': Arrow(t0, BOOL),
    'IS_UPSIDE_DOWN': Arrow(t0, BOOL),
    'IS_CHEESECAKE': Arrow(t0, BOOL),

    # Binary predicates
    'TOUCHING': Arrow(Arrow(t0, BOOL), Arrow(t0, BOOL)),
    'POINTING': Arrow(Arrow(t0, BOOL), Arrow(t0, BOOL)),
}

no_repetitions = set()
