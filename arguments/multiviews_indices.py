from typing import Dict, List, Literal

MULTIVIEW_INDICES: Dict[Literal[1, 3, 5], Dict[str, List[int]]] = {
    5: {  # 5 views
        "lego": [50, 59, 60, 70, 90],
        "hotdog": [0, 11, 23, 27, 37],
        "chair": [2, 25, 38, 79, 90],
        "drums": [2, 25, 38, 79, 90],
        "ficus": [2, 25, 38, 79, 90],
        "materials": [2, 25, 38, 79, 90],
        "mic": [2, 25, 38, 79, 90],
        "ship": [2, 25, 38, 79, 90],
    },
    3: {  # 3 views
        "lego": [50, 70, 90],
        "hotdog": [0, 23, 37],
        "chair": [2, 79, 90],
        "drums": [25, 38, 90],
        "ficus": [2, 79, 90],
        "materials": [2, 38, 79],
        "mic": [2, 38, 79],
        "ship": [2, 25, 79],
    },
    1: {  # single view
        "lego": [59],
        "hotdog": [2],
        "chair": [2],
        "drums": [3],
        "ficus": [2],
        "materials": [79],
        "mic": [25],
        "ship": [25],
    },
}
