import pygame

from environments.EnvironmentInterface import rescale_movement
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def convertDensityToRGB(density, low=4., high=6., channels=[True, True, True]):
    """
    Gets a value for the density of one box and returns the corresponding color

    :param density: density of the box (should be in range of the interval)
    :type density: float
    :param low: the minimum value for the d ensity
    :type low: float
    :param high: the maximum value for the density
    :type high: float
    :param channels: an array with 3 entries, where each entry says if the color channel should be used or not.
    e.g. if it's [True, False, True], we want to use the Red and Blue channel, but not the Green channel.
    :type channels: list[bool, bool, bool]
    :return: a RGB color
    :rtype: (int, int, int)
    """
    if not (low <= density <= high):
        raise AssertionError(f"Density {density} not in allowed range [{low},{high}]")

    if len(channels) != 3 or not all(type(channel) == bool for channel in channels):
        raise TypeError("Type of the channels array has to be a List of 3 Bool values.")

    total_number_of_colors = 256 ** sum(channels)
    # first normalize the density
    density = int(rescale_movement([low, high], density, [0., total_number_of_colors - 1]))

    RGB = [0, 0, 0]

    index = 0
    for i in reversed([i for i, boolean in enumerate(channels) if boolean]):
        RGB[i] = (density >> (index * 8)) & 255
        index += 1
    red, green, blue = RGB
    return red, green, blue


def convertRGBToDensity( RGB, low=4., high=6., channels=[True, True, True]):
    """
    Gets a RGB value of an box and returns the corresponding density of the box

    :param RGB: (red, green, blue) values
    :type RGB: (int, int, int)
    :param low: the minimum value for the density
    :type low: float
    :param high: the maximum value for the density
    :type high: float
    :return: density value
    :rtype: float
    """
    if not all(0 <= colorVal <= 255 for colorVal in RGB):
        raise AssertionError(f"RGB value {RGB} not allowed!")

    if len(channels) != 3 or (type(channels[i]) != bool for i in range(3)):
        raise TypeError("Type of the channels array has to be a List of 3 Bool values.")

    total_number_of_colors = 256 ** sum(channels)

    value = 0
    index = 0

    for i in reversed([i for i, boolean in enumerate(channels) if boolean]):
        value += RGB[i] * 256 ** index
        index += 1

    value /= total_number_of_colors

    # rescale the density
    density = rescale_movement([0., total_number_of_colors - 1], value, [low, high])

    return density


def convertDensityToGrayscale(density, low=4., high=6.):  # todo: fix
    """
    Gets a value for the density of one box and returns the corresponding grayscale value

    :param density: density of the box (should be in range of the interval)
    :type density: float
    :param low: the minimum value for the density
    :type low: float
    :param high: the maximum value for the density
    :type high: float
    :return: a RGB color
    :rtype: (int, int, int)
    """
    colormap = cm.gray
    norm = Normalize(vmin=low, vmax=high)
    red, green, blue, brightness = colormap(norm(density))

    return red, green, blue  # , brightness


def convertGrayscaleToDensity(RGB, low=4., high=6.):  # todo
    """
    Gets a Grayscale value of an box and returns the corresponding density of the box

    :param RGB: (red, green, blue) values
    :type RGB: (int, int, int)
    :param low: the minimum value for the density
    :type low: float
    :param high: the maximum value for the density
    :type high: float
    :return: density value
    :rtype: float
    """
    density = None
    return density

