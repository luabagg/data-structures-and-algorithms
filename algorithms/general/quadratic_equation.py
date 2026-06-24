import math
import cmath
import dataclasses
from typing import Iterable, Tuple

@dataclasses.dataclass
class FormulaCoefficients():
    a: int = 1
    b: int = 1
    c: int = 1

    def __str__(self) -> str:
        return f"{self.a} * x² + {self.b} * x + {self.c} = 0"

    def __iter__(self) -> Iterable[Tuple]:
        return ((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))

def get_formula_coefficients() -> FormulaCoefficients:
    fc = FormulaCoefficients()
    for c in fc:
        coefficient = c[0]
        try:
            coefficient_value = int(input(f"Please write the coefficient {coefficient}: "))
        except ValueError:
            print("Invalid number")
            exit()

        if coefficient == "a" and coefficient_value == 0:
            print("Coefficient a can't be 0")
            exit()

        setattr(fc, coefficient, coefficient_value)
    return fc

class QuadraticFormula():
    fc: FormulaCoefficients
    discriminant: int
    first_root: float|complex
    second_root: float|complex

    def __init__(self, fc: FormulaCoefficients) -> None:
        self.fc = fc
        self.discriminant = self.get_discriminant()
        self.calculate()

    def output(self):
        print("Equation:", self.fc)
        print("Discriminant:", self.discriminant)
        print("First Root:", self.first_root)
        print("Second Root:", self.second_root)

    def calculate(self):
        self.first_root = self.formula()
        self.second_root = self.formula(positive=False)

    def get_discriminant(self) -> float:
        return math.pow(self.fc.b, 2) - (4 * self.fc.a * self.fc.c)

    def square_root(self) -> float|complex:
        if self.discriminant >= 0:
            return math.sqrt(self.discriminant)
        return cmath.sqrt(self.discriminant)

    def formula(self, positive: bool = True) -> float:
        m = 1
        if positive is False:
            m = -1

        return (-self.fc.b + m * self.square_root()) / (2 * self.fc.a)

QuadraticFormula(
    get_formula_coefficients()
).output()
