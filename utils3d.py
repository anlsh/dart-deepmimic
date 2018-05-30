import numpy as np
import math

# Calculates Rotation Matrix given euler angles.
def eulerAnglesRotationAboutAngles(theta, axis_angles) :

    # Sourced from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.matmul(R_z, np.matmul( R_y, R_x ))
    R = np.matmul(R, eulerAnglesRotation(axis_angles))


def eulerAnglesRotation(theta) :

    # Sourced from https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])



    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])


    R = np.matmul(R_z, np.matmul( R_y, R_x ))
    return R

def get_transform_matrix(theta, translation):

    dimension = len(translation)
    matrix = np.zeros([dimension + 1] * 2)
    matrix[:,dimension] = np.append(translation, [1])
    matrix[:dimension, :dimension] = eulerAnglesRotation(theta)

    return matrix
