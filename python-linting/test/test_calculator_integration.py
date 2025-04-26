import pytest
from src.calculator import add, subtract, multiply, divide

def test_add_and_substract_integers():
    assert subtract(add(2, 3), 3) == 2

def test_multiply_and_divide_integers():
    assert divide(multiply(2, 3), 3) == 2


