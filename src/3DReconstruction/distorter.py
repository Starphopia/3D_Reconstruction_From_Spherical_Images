from typing import Tuple

class SamplingGridDistorter():
    def __init__(self, width : int, height : int, kernel_width : int, kernel_height : int\
                 stride : Tuple[int, int]):
        """
        :param width: width of the input image in pixels.
        :type width: integer
        :param height: height of the input image in pixels.
        :type height: integer
        :param kernel_height: height of the kernel in pixels.
        :type kernel_height: integer
        :param kernel_width: width of the kernel in pixels.
        :type kernel_width: integer
        :param stride: stride of the convolution filter.
        :param stride: 2-arity tuple of integers.
        """
        self.stride = stride
        self.delta_lat = np.pi / height
        self.delta_lon = 2 * np.pi / width
        self.W = width
        self.H = height
        self.KH = kernel_height
        self.KW = kernel_width
        
    def get_sampling_grid(self):
        """
        :return: the sampling grid distorted using the gnomonic projection 
        :rtype: numpy array of size (1, kernel height * image height, kernel width * image width, 2)
        """
        # Gets the latitudes and longitudes of the new sampling pattern.
        new_lons, new_lats = self._get_sampling_locations_radians()
        # Converts the latitudes and longitudes into pixel indices.
        new_xs, new_ys = self._get_sampling_locations_pixels(new_lons.transpose((1, 0, 2, 3)),\
                                                             new_lats.transpose((1, 0, 2, 3)))
    
        # Transposes the pixel indices so they are organised in rows not columns and puts them in the same array.
        grid = np.stack((new_ys, new_xs))
        # Matches x and y coordinates.
        grid = grid.transpose((1, 3, 2, 4, 0))
        
        # Reshapes the grid so that the sampling grid for each coordinate is placed next to each other.
        return grid.reshape(1, self.KH * self.H, self.KW * self.W, 2)
        
    def _make_kernel(self):
        """
        :return: The relative kernels for x and y separately.
        :rtype: 2-arity tuple of np arrays.
        """
        
        if self.KW % 2 == 0 or self.KH % 2 == 0:
            raise Exception("Sorry, even_sized kernels aren't supported yet.")
        
        relative_x = range(-(self.KW // 2), (self.KW // 2) + 1) 
        relative_y = range(-(self.KH // 2), (self.KH // 2) + 1)
        jj, kk = np.meshgrid(relative_x, relative_y)
        return np.tan(jj * self.delta_lon), np.tan(kk * self.delta_lat) / np.cos(kk * self.delta_lon)  

    
    def _get_sampling_locations_radians(self):        
        """
        :rtype: Tuple of two numpy arrays.
        :return: The longitudes and latitudes arrays.
        """
        kernel_xs, kernel_ys = self._make_kernel()

        xs = np.arange(0, self.W, self.stride[0])
        ys = np.arange(0, self.H, self.stride[1])
        lons = ((xs / self.W) - 0.5) * 2 * np.pi
        lats = ((ys / self.H) - 0.5) * np.pi
        
        rhos = np.sqrt(kernel_xs**2 + kernel_ys**2)
        
        # Replaces the 0s in rho with very small numbers to prevent division by zero error.
        rhos[rhos == 0] = 1e-8        
        nus = np.arctan(rhos)
        # Distorts the latitudes and longitudes.
        # Computes the distorted latitude for each row.
        new_lats = np.arcsin(np.array([np.cos(nus) * np.sin(lat) + ((kernel_ys * np.sin(rhos) * np.cos(lat)) / rhos)
                            for lat in lats]).clip(max=1, min=-1))
        # The latitude remains the same throughout the same row
        new_lats = np.array([new_lats for _ in lons])
        new_lons = np.array([np.arctan((kernel_xs * np.sin(nus)) 
                            / ((rhos * np.cos(lat) * np.cos(nus)) - (kernel_ys * np.sin(lat) * np.sin(nus))))
                            for lat in lats])
        new_lons = np.array([lon + new_lons for lon in lons])
        return new_lons, new_lats
        
    def _get_sampling_locations_pixels(self, new_lons, new_lats):
        """
        :rtype: Tuple of two numpy arrays.
        :return: the x and y pixel coordinates of the sampling locations.
        """
        x_coords = ((new_lons / (2 * np.pi) + 0.5) * self.W) % self.W
        y_coords = (new_lats / np.pi + 0.5) * self.H
        return x_coords, y_coords
    
    