def is_box_valid(x_min, y_min, x_max, y_max, img_width, img_height):
    if x_min < 0 or y_min < 0:
        return False
    if x_max > img_width or y_max > img_height:
        return False
    if x_max <= x_min or y_max <= y_min:
        return False
    return True
