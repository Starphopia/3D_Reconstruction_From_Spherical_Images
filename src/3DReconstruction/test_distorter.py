import math
import unittest
from distorter import Distorter
from numpy import testing

class TestDistorter(unittest.TestCase):
    def test_lon_and_lat_2d(self):
        # Simple cases
        simple_distorter = Distorter(200, 200, 3)
        testing.assert_almost_equal(simple_distorter.lon_and_lat_2d(0, 0), (-1*math.pi, math.pi*0.5))
        testing.assert_almost_equal(simple_distorter.lon_and_lat_2d(200, 200), (math.pi, math.pi*-0.5))
        testing.assert_almost_equal(simple_distorter.lon_and_lat_2d(100, 100), (0, 0))

        # More complex cases
        testing.assert_almost_equal(simple_distorter.lon_and_lat_2d(159, 32), ((118*math.pi)/200, (17/50)*math.pi))

        # Case with larger images
        large_distorter = Distorter(1778, 3687, 3)
        testing.assert_almost_equal(large_distorter.lon_and_lat_2d(20,1778), (-3.070915654, 0.05581077266))
        testing.assert_almost_equal(large_distorter.lon_and_lat_2d(1778, 3687), (math.pi, -0.5*math.pi))

        # Case with smaller images
        small_distorter = Distorter(10, 5, 3)
        testing.assert_almost_equal(small_distorter.lon_and_lat_2d(2, 1), (-0.6*math.pi, 0.3*math.pi))

    def test_lon_and_lat_3d(self):
        distorter = Distorter(200, 200, 3)
        # Where x is less than 0
        testing.assert_almost_equal(distorter.lon_and_lat_3d(-0.8830869300, 0.1722328441, 0.4387309164),
                                    (2.6804962137, 0.1730959393))
        # Where x is greater than 0
        testing.assert_almost_equal(distorter.lon_and_lat_3d(1, 0, 0), (0, 0))

        # Where x is 0
        testing.assert_almost_equal(distorter.lon_and_lat_3d(0, 1, 0), (math.pi / 2, math.pi / 2))

    def test_spherical_coords(self):
        distorter = Distorter(200, 200, 3)

        # Base cases
        testing.assert_almost_equal(distorter.spherical_coords(-1*math.pi, 0.5*math.pi),
                                (-7.50494368282489E-33,1,-6.1257422745431E-17))
        testing.assert_almost_equal(distorter.spherical_coords(math.pi, -0.5*math.pi),
                                (7.50494368282489E-33,-1,-6.1257422745431E-17))
        testing.assert_almost_equal(distorter.spherical_coords(0, 0), (0,0,1))

        # More complex cases
        testing.assert_almost_equal(distorter.spherical_coords(2, 3), (-0.90019763, 0.141120008, 0.411982246))
        testing.assert_almost_equal(distorter.spherical_coords(2.10424235, -1.034345098),
                                    (0.440078184, -0.859527828, -0.259890564))
        testing.assert_almost_equal(distorter.spherical_coords(0.0012, -0.0654),
                                    (0.001197434, -0.065353389, 0.997861464))

    def test_tangent_plane_vectors(self):
        distorter = Distorter(200, 200, 3)

        testing.assert_almost_equal(distorter.tangent_plane_vectors((-0.90019763, 0.141120008, 0.411982246)),
                                    ((0.4161468367, 0, 0.9092974267), (0.1283244670, 0.9900264961, -0.0581409345)), 3)
        testing.assert_almost_equal(distorter.tangent_plane_vectors((0.4400781840, -0.8595278280, -0.2598905640)),
                                    ((-0.508729312126641, 0, -0.860926528214316), (0.740114726674641, 0.511079223419921, -0.437067750752457)), 3)
        testing.assert_almost_equal(distorter.tangent_plane_vectors((0.0011974340, -0.0653533890, 0.9978614640)),
                                    ((0.9999992800, 0, -0.0011999994), (0.0000784247, 0.9978712838, 0.0652142232)), 3)

    def test_sampling_grid(self):
        distorter = Distorter(200, 200, 3)
        testing.assert_almost_equal(distorter.sampling_grid((0.4161468367, 0, 0.9092974267), (0.1283244670, 0.9900264961, -0.0581409345),
                                                            (-0.90019763, 0.141120008, 0.411982246)),
                                    [[[-0.917308330043916,0.110007171944807,0.385233575630315],
                                     [-0.913275571204027,0.141120008,0.383406423154603],
                                     [-0.909242812364138,0.172232844055192,0.381579270678892]],
                                     [[-0.904230388839889,0.110007171944807,0.413809398475712],
                                      [-0.90019763,0.141120008,0.411982246],
                                      [-0.896164871160111,0.172232844055192,0.410155093524288]],
                                     [[-0.891152447635862,0.110007171944807,0.442385221321108],
                                      [-0.887119688795973,0.141120008,0.440558068845397],
                                      [-0.883086929956084,0.172232844055192,0.438730916369685]]])
    def test_backpropogate(self):
        distorter = Distorter(200, 200, 3)
        testing.assert_almost_equal(distorter.backpropogate([[[-0.917308330043916,0.110007171944807,0.385233575630315],
                                                              [-0.913275571204027,0.141120008,0.383406423154603],
                                                              [-0.909242812364138,0.172232844055192,0.381579270678892]],
                                                             [[-0.904230388839889,0.110007171944807,0.413809398475712],
                                                              [-0.90019763,0.141120008,0.411982246],
                                                              [-0.896164871160111,0.172232844055192,0.410155093524288]],
                                                             [[-0.891152447635862,0.110007171944807,0.442385221321108],
                                                              [-0.887119688795973,0.141120008,0.440558068845397],
                                                              [-0.883086929956084,0.172232844055192,0.438730916369685]]]),
                                    [[[-137.344167567405,92.9825233327099], [-137.348124671513,90.9859317141242],
                                     [-137.352117296055,88.9803702498285]], [[-136.338560351643,92.9825233327099],
                                     [-136.338022757006,90.9859317141242], [-136.337480332421,88.9803702498285]],
                                     [[-135.332952063134,92.9825233327099], [-135.3279208425,90.9859317141242],
                                      [-135.322844470689,88.9803702498285]]])

if __name__ == '__main__':
    unittest.main()
