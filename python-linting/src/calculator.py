from typing import Union

number = Union[int, float]

def add(a: number, b: number) -> number:
    """Returns the sum of two numbers."""

    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers.")
    return a + b

def subtract(a: number, b: number) -> number:
    """Returns the difference between two numbers."""
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers.")
    return a - b

def multiply(a: number, b: number) -> number:
    """Returns the product of two numbers."""
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers.")
    return a * b

def divide(a: number, b: number) -> number:
    """
    Returns the division of a by b.
    Raises ValueError if b is zero.
    """
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers.")
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

