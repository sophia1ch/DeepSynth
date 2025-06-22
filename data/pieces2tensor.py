import torch

COLOR_MAP = {"red": 0, "blue": 1, "yellow": 2}
SHAPE_MAP = {"block": 0, "wedge": 1, "pyramid": 2}
ORIENTATION_MAP = {"upright": 0, "upside_down": 1, "flat": 2, "cheesecake": 3}

PAD_VECTOR = torch.tensor([7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1], dtype=torch.long)
MAX_OBJECTS = 7

def prolog_scene_to_tensor(scene_dict):
    """
    Convert a Prolog-generated scene (dict with pieces) into a tensor with shape (MAX_OBJECTS, 15)
    """
    rows = []
    id_to_index = {piece["id"]: i for i, piece in enumerate(scene_dict["pieces"])}
    num_pieces = len(scene_dict["pieces"])

    for piece in scene_dict["pieces"]:
        row = [piece["id"]]

        row.append(COLOR_MAP[piece["color"]])
        row.append(SHAPE_MAP[piece["shape"]])
        row.append(ORIENTATION_MAP[piece["orientation"]])

        # Touching in 6 directions
        touching = piece.get("touching", [])
        touching_ids = [7] * 6
        for rel in touching:
            direction = rel["direction"]
            target_id = rel["target"]
            touching_ids[direction] = id_to_index.get(target_id, 7)
        row.extend(touching_ids)

        # Pointing
        pointing_targets = piece.get("pointing", [])
        if pointing_targets:
            row.append(id_to_index.get(pointing_targets[0], 7))  # only use first pointing target
        else:
            row.append(7)

        # Bounding box
        bbox = piece.get("bbox", [-1, -1, -1, -1])
        row.extend(bbox)

        rows.append(torch.tensor(row, dtype=torch.long))

    # Pad to MAX_OBJECTS
    while len(rows) < MAX_OBJECTS:
        rows.append(PAD_VECTOR.clone())

    return torch.stack(rows)
