import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")

from EventTriggeredAntiCollision.trajectory_generation import interpolation

if __name__ == "__main__":
    breakpoints = [0, 1]  # Ende der Präditionsschritte
    Bernstein = interpolation.PiecewiseBernsteinPolynomial(
        breakpoints=breakpoints,
        dimension_trajectory=3,
        order=5)

    prediction_horizon = 1
    order = 5

    coeff_array = np.zeros((1, 5, 3))

    coeff_array[0, :, :] = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 2, 0], [2, 2, 0]])

    x_vector = np.linspace(0, 1, 50)

    coeffs = interpolation.TrajectoryCoefficients(coefficients=coeff_array,
                                                  valid=True, alternative_trajectory=None)
    pos = Bernstein.interpolate_vector(x_vector, coeffs)

    plt.scatter(coeff_array[0, :, 0], coeff_array[0, :, 1], c='b')
    plt.scatter(pos[:, 0], pos[:, 1], c='r')
    plt.grid()
    plt.show()

    print('Pause')
