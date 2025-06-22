import csv
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def normalize_task_name(s):
    return s.strip()

def normalize_program(s):
    s = s.strip()
    if s.endswith(" var0)"):
        s = s[:-6]  # remove ' var0)'
    elif s.endswith("var0)"):
        s = s[:-5]  # in case there's no space: 'var0)'
    return s + ")"

def normalize_for_comparison(s: str) -> str:
    s = s.strip()
    # if s.startswith("(") and s.endswith(")"):
    #     return s[1:-1].strip()
    return s

def normalize_for_rule_types(s: str) -> str:
    return s.strip()

def tokenize(s):
    # Turn the program string into a list of tokens
    return re.findall(r'\(|\)|[^\s()]+', s)

def extract_subprograms(expr):
    """Return [full_expr] for regular rules, or sub-rules for top-level AND/OR"""
    tokens = tokenize(expr)
    if not tokens or tokens[0] != '(':
        return [expr.strip()]
    
    # Get root func
    if len(tokens) > 1 and tokens[1] in ('AND', 'OR'):
        sub_exprs = []
        idx = 2  # Skip '(' and combinator
        while idx < len(tokens):
            if tokens[idx] == '(':
                depth = 1
                start = idx
                idx += 1
                while idx < len(tokens) and depth > 0:
                    if tokens[idx] == '(':
                        depth += 1
                    elif tokens[idx] == ')':
                        depth -= 1
                    idx += 1
                sub_exprs.append(' '.join(tokens[start:idx]))
            elif tokens[idx] == ')':
                break
            else:
                # Single token (like var0) — include it
                sub_exprs.append(tokens[idx])
                idx += 1
        return sub_exprs
    else:
        return [expr.strip()]
    

def extract_interaction_subprograms(expr):
    """
    Extract subprograms like (TOUCHING A B), (POINTING A B), etc.
    from inside the full DSL expression.
    """
    interaction_keywords = {"TOUCHING", "POINTING", "ON_TOP_OF"}
    tokens = tokenize(expr)
    idx = 0
    results = []

    def parse():
        nonlocal idx
        if tokens[idx] == '(':
            idx += 1
            if idx < len(tokens):
                func = tokens[idx]
                if func in interaction_keywords:
                    start = idx - 1  # include the opening '('
                    depth = 1
                    idx += 1
                    while idx < len(tokens) and depth > 0:
                        if tokens[idx] == '(':
                            depth += 1
                        elif tokens[idx] == ')':
                            depth -= 1
                        idx += 1
                    results.append(' '.join(tokens[start:idx]))
                else:
                    idx += 1
                    while idx < len(tokens) and tokens[idx] != ')':
                        parse()
                    if idx < len(tokens) and tokens[idx] == ')':
                        idx += 1
        else:
            idx += 1

    while idx < len(tokens):
        parse()
    return results

def normalize_symmetric_expr(expr: str) -> str:
    tokens = tokenize(expr)
    if len(tokens) < 3 or tokens[0] != '(':
        return expr.strip()

    op = tokens[1]
    symmetric_ops = {"AND", "OR", "TOUCHING", "POINTING", "ON_TOP_OF",
                     "AT_LEAST_2", "EXACTLY_2", "EVEN_2", "ODD_2"}

    if op not in symmetric_ops:
        return expr.strip()

    # Extract arguments
    args = []
    depth = 0
    current = []
    for t in tokens[2:-1]:  # Skip ( and )
        if t == '(':
            depth += 1
        elif t == ')':
            depth -= 1
        if depth == 0 and current:
            args.append(' '.join(current))
            current = []
        else:
            current.append(t)
        if depth == 0 and t not in ('(', ')'):
            args.append(t)

    if current:
        args.append(' '.join(current))

    # Normalize by sorting arguments
    args = [a.strip() for a in args if a.strip()]
    args.sort()
    return f"({op} {' '.join(args)})"

def collect_rule_types(expr):
    tokens = tokenize(expr)
    idx = 0
    rule_types = []

    known_rule_prefixes = {
        "AND", "OR",
        "AT_LEAST_1", "AT_LEAST_2", "AT_LEAST_INTERACTION",
        "EXACTLY_1", "EXACTLY_2", "EXACTLY_INTERACTION",
        "EVEN", "EVEN_1", "EVEN_2", "EVEN_INTERACTION",
        "ODD", "ODD_1", "ODD_2", "ODD_INTERACTION",
        "MORE_THAN", "EITHER_OR",
        "TOUCHING", "POINTING", "ON_TOP_OF",
        "IS_GROUNDED", "IS_RED", "IS_BLUE", "IS_YELLOW",
        "IS_BLOCK", "IS_PYRAMID", "IS_WEDGE",
        "IS_UPRIGHT", "IS_FLAT", "IS_UPSIDE_DOWN", "IS_CHEESECAKE",
        "IS_HORIZONTAL", "IS_VERTICAL"
    }

    def parse():
        nonlocal idx
        if tokens[idx] == '(':
            idx += 1
            if idx < len(tokens):
                func = tokens[idx]
                rule_types.append(func)
                idx += 1
                while idx < len(tokens) and tokens[idx] != ')':
                    parse()
                if idx < len(tokens) and tokens[idx] == ')':
                    idx += 1
        else:
            # Check if the token is a known rule primitive (atomic application like EVEN)
            if tokens[idx] in known_rule_prefixes:
                rule_types.append(tokens[idx])
            idx += 1

    while idx < len(tokens):
        parse()

    return rule_types


def evaluate_programs(csv_path):
    full_correct = 0
    full_incorrect = 0

    subrule_hits = defaultdict(int)
    subrule_totals = defaultdict(int)
    tracked_atomic_predicates = {
        "IS_RED", "IS_BLUE", "IS_YELLOW",
        "IS_BLOCK", "IS_PYRAMID", "IS_WEDGE",
        "IS_UPRIGHT", "IS_FLAT", "IS_UPSIDE_DOWN", "IS_CHEESECAKE",
        "IS_HORIZONTAL", "IS_VERTICAL"
    }

    subrule_totals_attr = defaultdict(int)
    subrule_hits_attr = defaultdict(int)

    # Count how many times each rule appears and how often it's correct
    rule_type_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

    with open(csv_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))

    current_task = None
    current_block = []
    start_index = 0

    def evaluate_block(block, task_name, start_index):
        nonlocal full_correct, full_incorrect

        norm_task = normalize_for_comparison(task_name)
        full_task = normalize_for_rule_types(task_name)
        rule_types = collect_rule_types(full_task)
        task_atoms = set(rt for rt in rule_types if rt in tracked_atomic_predicates)
        predictions = [normalize_program(row['program']) for row in block]
        for atom in task_atoms:
            subrule_totals_attr[atom] += 1

        # Collect all predicted rule types across predictions
        predicted_rule_types = set()
        for p in predictions:
            predicted_rule_types.update(collect_rule_types(p))

        for atom in task_atoms:
            if atom in predicted_rule_types:
                subrule_hits_attr[atom] += 1

        for r in rule_types:
            subrule_totals[r] += 1  # Always count appearances

        # Exact match
        norm_task_sym = normalize_symmetric_expr(norm_task)
        predictions_sym = [normalize_symmetric_expr(p) for p in predictions]
        if norm_task_sym in predictions_sym:
            print(f"✅ Task: {norm_task}\n   Full match found at block {start_index}")
            full_correct += 1
            for r in rule_types:
                subrule_hits[r] += 1
            return

        # Top-level combinator partial match
        task_tokens = tokenize(full_task)
        if len(task_tokens) > 1 and task_tokens[0] == '(' and task_tokens[1] in ('AND', 'OR'):
            combinator = task_tokens[1]
            task_subprograms = [normalize_for_comparison(s) for s in extract_subprograms(full_task)]

            for pred in predictions:
                pred_tokens = tokenize(pred)
                if len(pred_tokens) > 1 and pred_tokens[0] == '(' and pred_tokens[1] == combinator:
                    pred_subprograms = [normalize_for_comparison(s) for s in extract_subprograms(pred)]
                    task_subprograms_sym = [normalize_symmetric_expr(s) for s in task_subprograms]
                    pred_subprograms_sym = [normalize_symmetric_expr(s) for s in pred_subprograms]
                    matches = set(task_subprograms_sym) & set(pred_subprograms_sym)
                    if matches:
                        print(f"⚠️  Task: {norm_task}\n   Partial combinator match at block {start_index}")
                        for r in rule_types:
                            if any(r in collect_rule_types(p) for p in pred_subprograms):
                                subrule_hits[r] += 1
                        full_incorrect += 1
                        return

        # Interaction fallback
        task_interactions = [normalize_symmetric_expr(s) for s in extract_interaction_subprograms(full_task)]
        for pred in predictions:
            pred_interactions = [normalize_symmetric_expr(s) for s in extract_interaction_subprograms(pred)]
            matches = set(task_interactions) & set(pred_interactions)
            if matches:
                print(f"⚠️  Task: {norm_task}\n   Partial interaction match at block {start_index}")
                for r in rule_types:
                    if any(r in collect_rule_types(p) for p in pred_interactions):
                        subrule_hits[r] += 1
                full_incorrect += 1
                return

        # No match at all
        print(f"❌ Task: {norm_task}\n   No match in block {start_index}")
        full_incorrect += 1

    for i, row in enumerate(reader):
        task = row['task_name']
        if current_task is None:
            current_task = task
            current_block = [row]
            start_index = i
        elif task == current_task:
            current_block.append(row)
        else:
            evaluate_block(current_block, current_task, start_index)
            current_task = task
            current_block = [row]
            start_index = i

    if current_block:
        evaluate_block(current_block, current_task, start_index)

    print(f"\n✅ Full Matches: {full_correct}")
    print(f"❌ Full Failures: {full_incorrect}")
    print(f"📈 Full Program Accuracy: {(full_correct / (full_correct + full_incorrect)):.2%}")

    print("\n📊 Subrule Recognition Rate (per DSL primitive):")
    for rule_type in sorted(subrule_totals.keys()):
        hits = subrule_hits[rule_type]
        total = subrule_totals[rule_type]
        rate = hits / total if total > 0 else 0
        print(f"{rule_type:25}  🎯 {hits:3} / {total:3}  📈 {rate:.2%}")

    print("\n📊 Atomic Predicate Recognition Rate (all tasks, regardless of rule structure):")
    for rule_type in sorted(tracked_atomic_predicates):
        hits = subrule_hits_attr[rule_type]
        total = subrule_totals_attr[rule_type]
        rate = hits / total if total > 0 else 0
        print(f"{rule_type:25}  🎯 {hits:3} / {total:3}  📈 {rate:.2%}")

    # Prepare atomic predicate accuracy data
    atomic_predicates = sorted(tracked_atomic_predicates)
    acc_data = []
    for rule_type in atomic_predicates:
        hits = subrule_hits_attr[rule_type]
        total = subrule_totals_attr[rule_type]
        rate = hits / total if total > 0 else 0
        acc_data.append((rule_type, hits, total, rate))

    # Sort by accuracy
    acc_data.sort(key=lambda x: x[3], reverse=True)

    # Plot
    labels = [r[0] for r in acc_data]
    accuracies = [r[3] for r in acc_data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, accuracies, color=plt.cm.plasma([r[3] for r in acc_data]))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Atomic Predicate Recognition Accuracy (All Tasks)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zendo programs by task blocks.")
    parser.add_argument("--path", required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    evaluate_programs(args.path)