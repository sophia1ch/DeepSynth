from program import Function, BasicPrimitive

def dsl_to_prolog(func: Function) -> str:
    """Recursively convert a DSL Function into a Prolog rule string."""
    def unwrap(f):
        return dsl_to_prolog(f) if isinstance(f, Function) else str(f.name if hasattr(f, "name") else f)

    head = func.function
    args = func.arguments

    if head.primitive == 'IS_RED':
        return "red"
    if head.primitive == 'IS_BLUE':
        return "blue"
    if head.primitive == 'IS_YELLOW':
        return "yellow"
    if head.primitive == 'IS_BLOCK':
        return "block"
    if head.primitive == 'IS_WEDGE':
        return "wedge"
    if head.primitive == 'IS_PYRAMID':
        return "pyramid"
    if head.primitive == 'IS_UPRIGHT':
        return "upright"
    if head.primitive == 'IS_FLAT':
        return "flat"
    if head.primitive == 'IS_UPSIDE_DOWN':
        return "upside_down"

    if head.primitive.startswith("constant_"):
        return head.primitive.replace("constant_", "")

    # Count-based predicates
    if head.primitive in ['AT_LEAST_1', 'AT_LEAST_2', 'EXACTLY_1', 'EXACTLY_2']:
        count = unwrap(args[0])
        preds = [unwrap(a) for a in args[1:]]
        return f"{head.primitive.lower().split('_')[0]}({', '.join(preds)}, {count}, Structure)"

    if head.primitive in ['EVEN', 'ODD']:
        return f"{head.primitive.lower()}_number_of(Structure)"
    if head.primitive in ['EVEN_1', 'ODD_1']:
        pred = unwrap(args[0])
        return f"{head.primitive.lower().replace('_1', '_number_of')}({pred}, Structure)"
    if head.primitive in ['EVEN_2', 'ODD_2']:
        pred1 = unwrap(args[0])
        pred2 = unwrap(args[1])
        return f"{head.primitive.lower().replace('_2', '_number_of')}({pred1}, {pred2}, Structure)"

    if head.primitive in ['AND', 'OR']:
        left = unwrap(args[0])
        right = unwrap(args[1])
        return f"{head.primitive.lower()}([{left}, {right}])"

    if head.primitive in ['EITHER', 'EITHER_OR']:
        n1 = unwrap(args[0])
        n2 = unwrap(args[1])
        return f"either_or({n1}, {n2}, Structure)"

    if head.primitive in ['MORE_THAN']:
        p1 = unwrap(args[0])
        p2 = unwrap(args[1])
        return f"more_than({p1}, {p2}, Structure)"

    if head.primitive in ['AT_LEAST_INTERACTION', 'EXACTLY_INTERACTION', 'EVEN_INTERACTION', 'ODD_INTERACTION']:
        count = unwrap(args[0]) if head.primitive in ['AT_LEAST_INTERACTION', 'EXACTLY_INTERACTION'] else None
        interaction_func = args[1]
        inter_name = interaction_func.function.primitive.lower()
        pred1 = unwrap(interaction_func.arguments[0])
        pred2 = unwrap(interaction_func.arguments[1])
        pred_call = f"{pred1}, {pred2}, {inter_name}, Structure"
        if count:
            return f"{head.primitive.lower()}({pred_call}, {count})"
        return f"{head.primitive.lower()}({pred_call})"

    raise ValueError(f"Unsupported primitive in DSL: {head.primitive}")