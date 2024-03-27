import math
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
            coefficient_value = int(input("Please write the coefficient {coefficient}: "))
        except ValueError:
            print("Invalid number")
            exit()
        setattr(fc, coefficient, coefficient_value)
    return fc

class QuadraticFormula():
    fc: FormulaCoefficients
    discriminant: int
    first_root: int|None = None
    second_root: int|None = None

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
        if self.discriminant > 0:
            self.first_root = self.positive_formula()
            self.second_root = self.negative_formula()
        elif self.discriminant == 0:
            self.first_root, self.second_root = self.positive_formula()

    def get_discriminant(self) -> int:
        return math.pow(self.fc.b, 2) - (4 * self.fc.a * self.fc.c)

    def positive_formula(self) -> int:
        return (-self.fc.b + math.sqrt(self.discriminant)) / 2 * self.fc.a

    def negative_formula(self) -> int:
        return (-self.fc.b - math.sqrt(self.discriminant)) / 2 * self.fc.a

QuadraticFormula(
    get_formula_coefficients()
).output()
