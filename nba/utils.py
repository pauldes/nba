import yaml

def get_dict_from_yaml(yaml_file_path):
        """Get a YAML file as a dictionnary."""
        with open(yaml_file_path, "r") as stream:
            try:
                loaded = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Could not load YAML file", yaml_file_path, ":", e)
                return dict()
            else:
                return loaded