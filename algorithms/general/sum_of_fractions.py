"""
Algorithm for adding N fractions in the format a:b + c:d
"""

from __future__ import annotations
import re
import dataclasses
import math
from typing import List

@dataclasses.dataclass
class Fraction():
    numerator: int
    denominator: int

    def __str__(self):
        return f"{self.numerator}:{self.denominator}"

    def normalize(self, new_denominator: int):
        if self.denominator == new_denominator:
            return
        self.numerator *= new_denominator // self.denominator
        self.denominator = new_denominator

    def simplify(self):
        while True:
            greatest_common_divisor = math.gcd(self.numerator, self.denominator)

            if greatest_common_divisor == 1:
                break

            self.numerator //= greatest_common_divisor
            self.denominator //= greatest_common_divisor

def sum_fractions(fraction_classes: List[Fraction]) -> Fraction:
    new_denominator = 1
    for fc in fraction_classes:
        new_denominator *= fc.denominator

    for fc in fraction_classes:
        fc.normalize(new_denominator)

    total_sum = 0
    for fc in fraction_classes:
        total_sum += fc.numerator

    sum_fraction = Fraction(numerator=total_sum, denominator=new_denominator)
    sum_fraction.simplify()

    return sum_fraction

input_txt = "Write a fraction (n:d or Enter to end operation): "
user_input = ''
fraction_classes: List[Fraction] = []

while True:
    if user_input != "":
        input_txt += " + "

    user_input = input(input_txt)
    if user_input == "":
        break

    if re.search("\d:\d", user_input) is None:
        print("Invalid fraction")
        exit()

    splitted = user_input.split(":")
    fraction_classes.append(
        Fraction(
            numerator=int(splitted[0]),
            denominator=int(splitted[1])
        )
    )

    input_txt += f"{user_input}"

print(sum_fractions(fraction_classes))
