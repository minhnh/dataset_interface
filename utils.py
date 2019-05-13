from builtins import input      # for Python 2 compatibility
import argparse
import glob


class RawDescriptionAndDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Custom formatter for preserving both indentation and displaying default argument values"""
    pass


class TerminalColors(object):
    """https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python"""
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    @staticmethod
    def formatted_print(string, format):
        print(format + string + TerminalColors.ENDC)


def is_box_valid(x_min, y_min, x_max, y_max, img_width, img_height):
    if x_min < 0 or y_min < 0:
        return False
    if x_max > img_width or y_max > img_height:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True


def print_progress(current_index, total, prefix="", fraction=0.1):
    """
    Print progress every time the current_index match a multiple of a specified fraction of the total count
    For example, with the default fraction 0.1, the function would only print at 10%, 20%, and so on
    """
    increment = int(total * fraction)
    if increment == 0:
        # essentially print all indices if there area less than 10 entries in totalss
        increment = 1

    if current_index % increment == 0:
        print(prefix + "{}/{}".format(current_index, total))


def prompt_for_yes_or_no(promt_string, suffix=' [(y)es/(n)o]: ', blocking=True):
    """prompt for user [(y)es/(n)o] input, by default with append [(y)es/(n)o] to 'prompt_string'"""
    while True:
        # Note: input is
        reply = str(input(promt_string + suffix)).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        if not blocking:
            raise ValueError('invalid user input for yes/no prompt: ' + reply)
        print("invalid input: '{}', please retry".format(reply))


def case_insensitive_glob(pattern):
    """
    glob certain file types ignoring case

    :param pattern: file types (i.e. '*.jpg')
    :return: list of files matching given pattern
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))
