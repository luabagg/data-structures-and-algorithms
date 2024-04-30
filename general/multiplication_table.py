"""Prints a multiplication table of a given number"""

import math

def get_nums_len(numbers: tuple[int, ...]) -> tuple[int, ...]:
    """Returns a tuple containg the characteres of each integer

    Args:
        numbers (tuple[int, ...]): Tuple of numbers

    Returns:
        tuple[int, ...]: Tuple of the lengths of each number
    """
    return tuple(len(str(number)) for number in numbers)

try:
    TABLES = int(input("Enter the number of tables to be calculated: "))
    COLUMNS = int(input("Enter how many columns to display: "))
except ValueError:
    print("Invalid input")

MAX_MULTIPLIER = 10
"""The max multiplier of each table"""
MAX_MULTIPLICAND = TABLES - 1
"""The maximum multiplicand. -1 because it starts at index 0"""

TABLE_SPACING = " " * 3

MAX_CHARS_SIZE = sum(get_nums_len((MAX_MULTIPLIER, MAX_MULTIPLICAND, MAX_MULTIPLICAND * MAX_MULTIPLIER)))
MAX_TEXT_SIZE = MAX_CHARS_SIZE + 10
"""The maximum text size. +10 sums the chars len of the formatting"""

def get_header_line(it: int) -> str:
    header_lines = "-" * MAX_TEXT_SIZE

    return f"{header_lines}{TABLE_SPACING}"

def get_header_title(it: int) -> str:
    text = f"Number {it} table"

    total_spaces = MAX_TEXT_SIZE - len(text)
    pre = " " * math.floor(total_spaces / 2)
    suf = pre
    if total_spaces % 2 != 0:
        suf += " "

    return f"{pre}{text}{suf}{TABLE_SPACING}"

def print_headers_row(starting_table: int) -> None:
    table_definition = [
        get_header_line,
        get_header_title,
        get_header_line
    ]

    for item in table_definition:
        text = ""
        finish_table = starting_table + COLUMNS
        for table_number in range(starting_table, finish_table):
            if table_number == TABLES:
                break
            text += item(table_number)
        print(text)

def get_line_item(table_number: int, mult: int) -> str:
    result = table_number * mult

    loop_chars_size = sum(get_nums_len((table_number, mult, result)))
    format_spaces = " " * (MAX_CHARS_SIZE - loop_chars_size)

    return f"| {table_number} X {mult} = {result} {format_spaces}|{TABLE_SPACING}"

def print_tables_row(number: int):
    for mult in range(0, MAX_MULTIPLIER + 1):
        text = ""
        for table_number in range(number, number + COLUMNS):
            if table_number == TABLES:
                break
            text += get_line_item(table_number, mult)
        print(text)

def main():
    """
    Prints each line determined by the number of columns
    """
    for starting_table in range(0, TABLES, COLUMNS):
        if starting_table > TABLES:
            break
        print_headers_row(starting_table)
        print_tables_row(starting_table)

if __name__ == "__main__":
    main()
