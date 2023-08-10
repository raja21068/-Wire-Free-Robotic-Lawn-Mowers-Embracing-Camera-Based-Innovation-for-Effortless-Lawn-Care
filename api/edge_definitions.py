det_classes = [
    'potted plant', 'person', 'animal', 'bench', 'table', 'kite', 'vehicle', 'toy', 'frisbee', 'motorcycle',
    'bicycle', 'chair', 'bottle', 'fire hydrant', 'rock', 'construction', 'trashcan', 'plant', 'tire',
    'others', 'leaf debris', 'hedgehog', 'faeces', 'trashcan lid', 'sprinkler', 'branch'
]
seg_classes = [
    'lawn', 'road', 'terrain', 'sky', 'person', 'animal', 'toy', 'leaf_debris', 'plant', 'vehicle',
    'construction', 'fire_hydrant', 'sprinkler', 'bench', 'table', 'chair', 'pipe', 'faeces', 'rock',
    'bottle', 'trashcan_lid', 'others'
]
seg_soft_classes = ['lawn', 'road']
seg_soft_class_ids = [1, 2]
seg_class_ids = {
    'lawn': 1,          'road': 2,          'terrain': 3,       'sky': 4,       'person': 5,
    'animal': 6,        'toy': 7,           'leaf_debris': 8,   'plant': 9,     'vehicle': 10,
    'construction': 11, 'fire_hydrant': 12, 'sprinkler': 13,    'bench': 14,    'table': 15,
    'chair': 16,        'pipe': 17,         'faeces': 18,       'rock': 19,     'bottle': 20,
    'trashcan_lid': 21,
    'others': 255
}
seg_id_class = {
    1: 'lawn', 2: 'road', 3: 'terrain', 4: 'sky', 5: 'person', 6: 'animal', 7: 'toy', 8: 'leaf_debris',
    9: 'plant', 10: 'vehicle', 11: 'construction', 12: 'fire_hydrant', 13: 'sprinkler', 14: 'bench',
    15: 'table', 16: 'chair', 17: 'pipe', 18: 'faeces', 19: 'rock', 20: 'bottle', 21: 'trashcan_lid',
    255: 'others'
}
edge_pixels = {"soft_edge": 0, "hard_edge": 1, "unknown": 2}
edge_color = [[0, 192, 96], [0, 128, 160], [128, 0, 96], [0, 32, 192]]
edge_colors = [[32, 223, 128], [0, 0, 255], [0, 255, 0]]

red_colors = [(50, 50, 250), (200, 200, 250), (0, 0, 255), (0, 100, 255), (0, 200, 255)]  # Red
green_colors = [(50, 255, 50), (200, 255, 200), (0, 255, 0), (0, 255, 100), (0, 255, 200)]  # Green
blue_colors = [(255, 50, 50), (255, 100, 100), (255, 0, 0), (255, 100, 0), (255, 200, 0)]  # Blue

hard_edge_colors = red_colors
soft_edge_colors = green_colors
