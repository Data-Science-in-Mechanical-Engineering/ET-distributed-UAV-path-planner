import numpy as np


class Cuboid:
    """ This class represents a cuboid which is used to define the boundaries of the drone testbed """

    def __init__(self, origin, v100, v010, v001):
        self.__origin = origin
        self.__v100 = v100
        self.__v010 = v010
        self.__v001 = v001

        self.__faces = []
        self.__faces.append(Face(self.__v100, self.__v010, self.__origin))
        self.__faces.append(Face(self.__v100, self.__v010, self.__origin + self.__v001))
        self.__faces.append(Face(self.__v100, self.__v001, self.__origin))
        self.__faces.append(Face(self.__v100, self.__v001, self.__origin + self.__v010))
        self.__faces.append(Face(self.__v010, self.__v001, self.__origin))
        self.__faces.append(Face(self.__v010, self.__v001, self.__origin + self.__v100))


    def edges(self):
        """ returns the relevant edges of the cuboid"""

        return (self.__origin.tolist(), (self.__origin + self.__v001 + self.__v010 + self.__v100).tolist())

    def center(self):
        """ returns the center of the cuboid"""

        return self.__origin + 0.5 * (self.__v100 + self.__v010 + self.__v001)

    @property
    def origin(self):
        """ return origin of cuboid"""

        return self.__origin

    def calc_min_dist_to_face(self, p):
        """ calculates the minimum distance between p and the faces of the cuboid

        Parameters:
            p: np.array
        point for which the distance should be calculated"""

        dist = []
        for face in self.__faces:
            dist.append(face.calc_dist_to_face(p))

        return np.min(dist)

    def calc_min_dist_vector(self, p):
        """ calculates the vector between p and the closest face of self

        Parameters:
            p: np.array
        point to which the distance vector should be calculated"""

        # calculate distance of faces to p
        dist = []
        for face in self.__faces:
            dist.append(face.calc_dist_to_face(p))

        # find closest face
        idx = np.array(dist).argmin()
        # calculate vector between face and p
        dist_vec = self.__faces[idx].calc_vector_to_p(p)

        return dist_vec


class Face:
    """ This class represents a face of a cuboid"""

    def __init__(self, v1, v2, origin):
        self.__origin = origin
        self.__v1 = v1
        self.__v2 = v2
        self.__normal_vector = np.cross(v1, v2)
        self.__a = np.dot(self.__normal_vector, self.__origin)

    def calc_dist_to_face(self, p):
        """ calculates the distance to this face

        Parameters:
            p: point to which the distance should be calculated"""

        dist = abs(np.dot(self.__normal_vector, p) - self.__a) / np.linalg.norm(self.__normal_vector, ord=2)

        return dist

    def calc_vector_to_p(self, p):
        """ calculates the normal vector on the face to the point p"""

        # calculate lenght distance vector
        dist = (np.dot(self.__normal_vector, p) - self.__a) / np.linalg.norm(self.__normal_vector, ord=2)

        # calculate distance * normalized normal vector
        return self.__normal_vector / np.linalg.norm(self.__normal_vector, ord=2) * dist
