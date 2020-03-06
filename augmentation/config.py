import yaml

class ConfigParams(object):
    # appearance parameters
    prob_rand_color = 0.
    prob_rand_noise = 0.
    prob_rand_bright = 0.
    bright_shift_range = None

    prob_rand_rotation = 0.

    # scaling parameters
    min_scale = 0.
    max_scale = 0.
    margin = 0.03
    max_obj_size_in_bg = 0.

    morph_kernel_size = 0
    morph_iter_num = 0

def read_config_params(config_file_path):
    '''Reads augmentation configuration parameters from the given configuration file.
    Returns a ConfigParams object with the parameters.

    Expects the following parameters to be defined in the config file:
    prob_rand_color, prob_rand_noise, prob_rand_bright, bright_shift_range,
    prob_rand_rotation, min_scale, max_scale, margin, max_obj_size_in_bg,
    morph_kernel_size, morph_iter_num

    Throws a KeyError if any of these parameters are undefined.

    Throws a TypeError if the value at 'bright_shift_range' is not an iterable.

    Keyword arguments:
    config_file_path: str -- path to an augmentation configuration file

    '''
    config_params = ConfigParams()
    params_dict = None
    with open(config_file_path, 'r') as config_file:
        params_dict = yaml.load(config_file)

    config_params.prob_rand_color = params_dict['prob_rand_color']
    config_params.prob_rand_noise = params_dict['prob_rand_noise']
    config_params.prob_rand_bright = params_dict['prob_rand_bright']
    config_params.bright_shift_range = tuple(params_dict['bright_shift_range'])

    config_params.prob_rand_rotation = params_dict['prob_rand_rotation']

    config_params.min_scale = params_dict['min_scale']
    config_params.max_scale = params_dict['max_scale']
    config_params.margin = params_dict['margin']
    config_params.max_obj_size_in_bg = params_dict['max_obj_size_in_bg']

    config_params.morph_kernel_size = params_dict['morph_kernel_size']
    config_params.morph_iter_num = params_dict['morph_iter_num']

    return config_params
