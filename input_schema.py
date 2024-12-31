INPUT_SCHEMA = {
    "image_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://github.com/microsoft/TRELLIS/raw/main/assets/example_image/T.png"]
    },
    "geometry_seed": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [42]
    },
    "sparse_structure_steps": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [20]
    },
    "sparse_structure_strength": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [7.5]
    },
    "slat_steps": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [20]
    },
    "slat_strength": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [3.0]
    },
    "simplify": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [0.95]
    },
    "texture_size": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [1024]
    },
    "timeout": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [300]
    }
    
}
