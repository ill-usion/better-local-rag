import yaml

def load_options(path: str) -> any:
    ''' Loads an options file '''

    with open(path, "r") as f:
        options = yaml.safe_load(f)
    
    return options
