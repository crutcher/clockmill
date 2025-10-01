#!/usr/bin/env python3

# This is a fluid simulator using the lattice Boltzmann method.
# Using D2Q9 and peiodic boundary, and used no external library.
# It generates two ripples at 50,50 and 50,40.
# Reference: Erlend Magnus Viggen's Master thesis, "The Lattice Boltzmann Method with Applications in Acoustics".
# For Wikipedia under CC-BY-SA license.

import math

# Maximum solving steps
MaxSteps = 120

# Resolution of the simulation
GridShape = 100
# The speed of sound, specifically 1/sqrt(3) ~ 0.57
SpeedOfSound = 1 / math.sqrt(3)
# time relaxation constant
TimeRelaxationConstant = 0.5


# Weights in D2Q9
Weights = [1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36]

# Discrete velocity vectors
DiscreteVelocityVectors = [
    [-1, 1],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
]


# A Field2D class
class Field2D:
    field: list[list[list[int]]]
    res: int

    def __init__(self, res: int) -> None:
        self.field = [
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0]
                for _ in range(res)
            ]
            for _ in range(res)
        ]
        self.res = res

    # This visualizes the simulation, can only be used in a terminal
    @staticmethod
    def VisualizeField(a, sc, res) -> None:
        for h in range(res):
            row = ""
            for w in range(res):
                y = int(h * a.res / res)
                x = int(w * a.res / res)

                flowmomentem = a.Momentum(y, x)
                col = "\033[38;2;{0};{1};{2}m██".format(
                    int(127 + sc * flowmomentem[0]), int(127 + sc * flowmomentem[1]), 0
                )
                row = row + col
            print(row)

    def print(self) -> None:
        Field2D.VisualizeField(self, 128, self.res)

    # Momentum of the field
    def Momentum(self, y, x) -> tuple[float, float]:
        v = sum(self.field[y][x])
        return [
            v * VelocityField[y][x][u]
            for u in [0, 1]
        ]


a = Field2D(GridShape)

# The velocity field: H, W, J:9, U:2
VelocityField: list[list[list[float]]] = [
    [[0.0, 0.0] for _ in range(GridShape)] for _ in range(GridShape)
]

# The density field
DensityField: list[list[float]] = [
    [1.0 for _ in range(GridShape)] for _ in range(GridShape)
]

# Set initial condition
DensityField[50][50] = 2.0
DensityField[40][50] = 2.0

# Solve
for s in range(MaxSteps):
    # Collision Step
    df = Field2D(GridShape)
    for h in range(GridShape):
        for w in range(GridShape):
            # The Flow Velocity
            FlowVelocity = VelocityField[h][w]

            # The current density
            density = DensityField[h][w]

            for j in range(9):
                Velocity = a.field[h][w][j]
                FirstTerm = Velocity

                Dotted = sum(
                    FlowVelocity[u] * DiscreteVelocityVectors[j][u]
                    for u in [0, 1]
                )

                # #The taylor expansion of equilibrium term
                taylor = (
                        1
                        + (Dotted / (SpeedOfSound ** 2))
                        + ((Dotted ** 2) / (2 * SpeedOfSound ** 4))
                        - (
                                (FlowVelocity[0] ** 2 + FlowVelocity[1] ** 2)
                                / (2 * SpeedOfSound ** 2)
                        )
                )
                # The equilibrium
                equilibrium = density * taylor * Weights[j]
                SecondTerm = (equilibrium - Velocity) / TimeRelaxationConstant
                df.field[h][w][j] = FirstTerm + SecondTerm

    # Streaming Step
    for h in range(GridShape):
        for w in range(GridShape):
            for j in range(9):
                # Target, the lattice point this iteration is solving
                TargetY = h + DiscreteVelocityVectors[j][1]
                TargetX = w + DiscreteVelocityVectors[j][0]
                # Periodic Boundary
                if TargetY == GridShape and TargetX == GridShape:
                    a.field[TargetY - GridShape][TargetX - GridShape][j] = df.field[h][w][j]
                elif TargetX == GridShape:
                    a.field[TargetY][TargetX - GridShape][j] = df.field[h][w][j]
                elif TargetY == GridShape:
                    a.field[TargetY - GridShape][TargetX][j] = df.field[h][w][j]
                elif TargetY == -1 and TargetX == -1:
                    a.field[TargetY + GridShape][TargetX + GridShape][j] = df.field[h][w][j]
                elif TargetX == -1:
                    a.field[TargetY][TargetX + GridShape][j] = df.field[h][w][j]
                elif TargetY == -1:
                    a.field[TargetY + GridShape][TargetX][j] = df.field[h][w][j]
                else:
                    a.field[TargetY][TargetX][j] = df.field[h][w][j]

    # Calculate macroscopic variables
    for h in range(GridShape):
        for w in range(GridShape):
            # Recompute Density Field
            DensityField[h][w] = sum(a.field[h][w])

            # Recompute Velocity Field
            VelocityField[h][w] = [
                sum(
                    DiscreteVelocityVectors[j][u] * a.field[h][w][j]
                    for j in range(9)
                ) / DensityField[h][w]
                for u in [0, 1]
            ]

    # Visualize
    a.print()
