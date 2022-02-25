import math
from msilib.schema import Error
from multiprocessing.sharedctypes import Value
import numpy as np

class Distorter:
    """ Given a pixel's coordinates, calculate the distortion based on the
    height and width of the input image. """

    def __init__(self, width, height, conv_size):
        ''' Stores the height and width of the input image and the size of
        the convolution kernel.'''

        # try:
        if width <= 0 or height <= 0 or conv_size < 0:
            raise LessThanZeroException
        self.width = width
        self.height = height

        if conv_size > width or conv_size > height:
            raise TooBigKernelException

        self.kernel_size = conv_size
        self.R = self.relative_sampling_positions(conv_size)
        # except TooBigKernelException:
        #     print("The size of the kernel must not be larger than the image.")
        # except LessThanZeroException:
        #     print("The widths and the heights of the image and the kernel can't be zero.")

    def relative_sampling_positions(self, size):
        """ Computes the relative sampling positions as a matrix given the
        required size of the square convolution kernel. Assumes that the size
        of the kernel is odd so that there is a center. """

        try:
            if size % 2 == 0 or size < 1:
                raise ValueError()

            upper_lim = size // 2
            lower_lim = -1 * upper_lim

            return np.array([[(i, j) for j in range(lower_lim, upper_lim + 1)]
                             for i in range(lower_lim, upper_lim + 1)])
        except OddNumbersOnlyException:
            print("Invalid size: {size}. The size must be a positive odd number.")

    def lon_and_lat_2d(self, x, y):
        """ Returns the latitude and longitude of given coordinates. """
        lon = (x -(self.width / 2)) * ((2 * math.pi) / self.width)
        lat = (self.height*0.5 - y) * (math.pi / self.height)
        return lon, lat


    def lon_and_lat_3d(self, x, y, z):
        """ Backprojects, obtaining a latitude and longitude from the
        spherical domain. """
        lon = 0
        if x == 0:
            lon = math.pi / 2
        elif x > 0:
            lon = math.atan(z / x)
        else:
            lon = math.atan(z / x) + math.pi

        lat = math.asin(y)
        return lon, lat

    def spherical_coords(self, lon, lat):
        """ Returns the spherical coordinates of a given latitude and
        longitude. """
        xu = math.cos(lat) * math.sin(lon)
        yu = math.sin(lat)
        zu = math.cos(lat) * math.cos(lon)

        return np.array([xu, yu, zu])

    def tangent_plane_vectors(self, spherical_coords):
        """ Computes the direction vectors that define a tangent plane to the
        given spherical point, tx and ty. spherical_coords is a numpy array. """
        spherical_coords = np.array(spherical_coords)
        tx = self.normalize(np.cross(np.array([0, 1, 0]), spherical_coords))
        ty = self.normalize(np.cross(spherical_coords, tx))
        return tx, ty

    def normalize(self, vector):
        """ Takes in a vector and converts it into a unit vector with the same
        direction. """
        norm=np.linalg.norm(vector, ord=2)
        if norm==0:
            norm=np.finfo(vector.dtype).eps
        return vector/norm


    def sampling_grid(self, tx, ty, spherical_coords):
        """ Computes the sampling grid on the tangent plane. """

        # Spatial resolution - distance between elements
        res = math.tan((2 * math.pi) / self.width)

        # Computes the sampling grid locations.
        sampling_grid_locations = np.empty(shape=(self.R.shape[0],
                                                  self.R.shape[1], 3))
        for i in range(0, self.R.shape[0]):
            for j in range(0, self.R.shape[1]):
                r = self.R[i][j]
                tx = np.array(tx)
                ty = np.array(ty)
                sampling_grid_locations[i, j] = spherical_coords + (res * (tx * r[0] + ty * r[1]))
        return sampling_grid_locations

    def backpropogate(self, sampling_grid_locations):
        """ Backpropogates point locations from the sampling grid to the
        equirectangular domain. """
        # Converts the sampling grid locations into a numpy array if it isn't already.
        if not isinstance(sampling_grid_locations, np.ndarray):
            sampling_grid_locations = np.array(sampling_grid_locations)

        # Array storing the new x, y sampling positions on the 2d plane.
        equirect = np.zeros(shape=(sampling_grid_locations.shape[0],
                                   sampling_grid_locations.shape[1], 2))
        for i in range(0, sampling_grid_locations.shape[0]):
            for j in range(0, sampling_grid_locations.shape[1]):
                point = sampling_grid_locations[i][j]
                lon, lat = self.lon_and_lat_3d(point[0], point[1], point[2])
                new_x = ((lon / (2 * math.pi)) + 0.5) * self.width
                new_y = (0.5 - (lat / math.pi)) * self.height
                equirect[i][j] = [new_x, new_y]

        return equirect

    def distort(self, x, y):
        """ Distorts the sampling locations on the equirectangular plane. """
        lon, lat = self.lon_and_lat_2d(x, y)
        sphere_coords = self.spherical_coords(lon, lat)
        tx, ty = self.tangent_plane_vectors(sphere_coords)
        sampling_grid = self.sampling_grid(tx, ty, sphere_coords)
        return self.backpropogate(sampling_grid)



class TooBigKernelException(Exception):
    pass

class LessThanZeroException(Exception):
    pass

class OddNumbersOnlyException(Exception):
    pass

