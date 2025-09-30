# This is a fluid simulator using the lattice Boltzmann method.
# Using D2Q9 and peiodic boundary, and used no external library.
# It generates two ripples at 50,50 and 50,40.
# Reference: Erlend Magnus Viggen's Master thesis, "The Lattice Boltzmann Method with Applications in Acoustics".
# For Wikipedia under CC-BY-SA license.

import math

# Define some utilities
def sum(a):
    s = 0
    for e in a:
        s = s + e
    return s


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
    def __init__(self, res: int):
        self.field = []
        for b in range(res):
            fm = []
            for a in range(res):
                fm.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
            self.field.append(fm[:])
        self.res = res

    # This visualize the simulation, can only be used in a terminal
    @staticmethod
    def VisualizeField(a, sc, res):
        stringr = ""
        for u in range(res):
            row = ""
            for v in range(res):
                n = int(u * a.res / res)
                x = int(v * a.res / res)
                flowmomentem = a.Momentum(n, x)
                col = "\033[38;2;{0};{1};{2}m██".format(
                    int(127 + sc * flowmomentem[0]), int(127 + sc * flowmomentem[1]), 0
                )
                row = row + col
            print(row)
            stringr = stringr + row + "\n"
        return stringr

    # Momentum of the field
    def Momentum(self, x, y):
        return velocityField[y][x][0] * sum(self.field[y][x]), velocityField[y][x][
            1
        ] * sum(self.field[y][x])


# Resolution of the simulation
res = 100
a = Field2D(res)
# The velocity field
velocityField = []
for DummyVariable in range(res):
    DummyList = []
    for DummyVariable2 in range(res):
        DummyList.append([0, 0])
    velocityField.append(DummyList[:])
# The density field
DensityField = []
for DummyVariable in range(res):
    DummyList = []
    for DummyVariable2 in range(res):
        DummyList.append(1)
    DensityField.append(DummyList[:])
# Set initial condition
DensityField[50][50] = 2
DensityField[40][50] = 2
# Maximum solving steps
MaxSteps = 120
# The speed of sound, specifically 1/sqrt(3) ~ 0.57
SpeedOfSound = 1 / math.sqrt(3)
# time relaxation constant
TimeRelaxationConstant = 0.5
# Solve
for s in range(MaxSteps):
    # Collision Step
    df = Field2D(res)
    for y in range(res):
        for x in range(res):
            for v in range(9):
                Velocity = a.field[y][x][v]
                FirstTerm = Velocity
                # The Flow Velocity
                FlowVelocity = velocityField[y][x]
                Dotted = (
                    FlowVelocity[0] * DiscreteVelocityVectors[v][0]
                    + FlowVelocity[1] * DiscreteVelocityVectors[v][1]
                )
                # #The taylor expainsion of equilibrium term
                taylor = (
                    1
                    + ((Dotted) / (SpeedOfSound**2))
                    + ((Dotted**2) / (2 * SpeedOfSound**4))
                    - (
                        (FlowVelocity[0] ** 2 + FlowVelocity[1] ** 2)
                        / (2 * SpeedOfSound**2)
                    )
                )
                # The current density
                density = DensityField[y][x]
                # The equilibrium
                equilibrium = density * taylor * Weights[v]
                SecondTerm = (equilibrium - Velocity) / TimeRelaxationConstant
                df.field[y][x][v] = FirstTerm + SecondTerm
    # Streaming Step
    for y in range(0, res):
        for x in range(0, res):
            for v in range(9):
                # Target, the lattice point this iteration is solving
                TargetY = y + DiscreteVelocityVectors[v][1]
                TargetX = x + DiscreteVelocityVectors[v][0]
                # Peiodic Boundary
                if TargetY == res and TargetX == res:
                    a.field[TargetY - res][TargetX - res][v] = df.field[y][x][v]
                elif TargetX == res:
                    a.field[TargetY][TargetX - res][v] = df.field[y][x][v]
                elif TargetY == res:
                    a.field[TargetY - res][TargetX][v] = df.field[y][x][v]
                elif TargetY == -1 and TargetX == -1:
                    a.field[TargetY + res][TargetX + res][v] = df.field[y][x][v]
                elif TargetX == -1:
                    a.field[TargetY][TargetX + res][v] = df.field[y][x][v]
                elif TargetY == -1:
                    a.field[TargetY + res][TargetX][v] = df.field[y][x][v]
                else:
                    a.field[TargetY][TargetX][v] = df.field[y][x][v]
    # Calculate macroscopic variables
    for y in range(res):
        for x in range(res):
            # Recompute Density Field
            DensityField[y][x] = sum(a.field[y][x])
            # Recompute Flow Velocity
            FlowVelocity = [0, 0]
            for DummyVariable in range(9):
                FlowVelocity[0] = (
                    FlowVelocity[0]
                    + DiscreteVelocityVectors[DummyVariable][0]
                    * a.field[y][x][DummyVariable]
                )
            for DummyVariable in range(9):
                FlowVelocity[1] = (
                    FlowVelocity[1]
                    + DiscreteVelocityVectors[DummyVariable][1]
                    * a.field[y][x][DummyVariable]
                )
            FlowVelocity[0] = FlowVelocity[0] / DensityField[y][x]
            FlowVelocity[1] = FlowVelocity[1] / DensityField[y][x]
            # Insert to Velocity Field
            velocityField[y][x] = FlowVelocity
    # Visualize
    Field2D.VisualizeField(a, 128, 100)