import yaml
import sys

def yaml_to_protobuf_converter(yml_filename, protobuf_filename):
    """
    Function to parse a yaml file containing id and classes into a labelMap in
    protobuf format.

    args:
    (1) yml_filename: Name of the input yaml file
    (2) protobuf_filename:  Name of the output protobuf file
    """

    # Read YAML file
    with open(yml_filename, 'r') as yml_file:
        classes_dict = yaml.load(yml_file, Loader=yaml.FullLoader)

    f = open(protobuf_filename, 'w')

    for id, class_name in classes_dict.items():
        f.write("item {\n")
        f.write("  id: {}\n".format(id))
        f.write("  name: '{}'\n".format(class_name))
        f.write("}\n")
        f.write('\n')

    f.close()


if __name__ == '__main__':
    """
    args:
    (1) yml_filename: Name of the input yaml file
    (2) protobuf_filename:  Name of the output protobuf file
    """

    yml_filename = sys.argv[1]
    protobuf_filename = sys.argv[2]

    yaml_to_protobuf_converter(yml_filename, protobuf_filename)
