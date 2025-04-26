import pytest
from src.calculator import add, subtract, multiply, divide

def test_add_integers():
    """Test adding two positive integers."""
    assert add(2, 3) == 5

def test_add_floats():
    """Test adding two floating-point numbers."""
    assert add(2.5, 3.5) == 6.0

def test_add_negative():
    """Test adding positive and negative numbers."""
    assert add(-5, 3) == -2

def test_add_type_error():
  """Test that add raises TypeError for non-numeric input."""
  with pytest.raises(TypeError):
    add("a", 3)
  with pytest.raises(TypeError):
    add(5, "b")

def test_subtract_integers():
  """Test subtracting two positive integers."""
  assert subtract(5, 2) == 3

def test_subtract_floats():
  """Test subtracting two floating-point numbers."""
  assert subtract(5.5, 2.1) == pytest.approx(3.4) # Use approx for float comparisons

def test_subtract_negative():
  """Test subtracting with negative numbers."""
  assert subtract(-5, -3) == -2
  assert subtract(3, -2) == 5

def test_subtract_type_error():
  """Test that subtract raises TypeError for non-numeric input."""
  with pytest.raises(TypeError):
    subtract("a", 3)
  with pytest.raises(TypeError):
    subtract(5, "b")

def test_multiply_integers():
  """Test multiplying two positive integers."""
  assert multiply(3, 4) == 12

def test_multiply_floats():
  """Test multiplying two floating-point numbers."""
  assert multiply(2.5, 4.0) == 10.0

def test_multiply_by_zero():
  """Test multiplying by zero."""
  assert multiply(5, 0) == 0
  assert multiply(0, 5) == 0

def test_multiply_type_error():
  """Test that multiply raises TypeError for non-numeric input."""
  with pytest.raises(TypeError):
    multiply("a", 3)
  with pytest.raises(TypeError):
    multiply(5, "b")

def test_divide_integers():
  """Test dividing two positive integers."""
  assert divide(10, 2) == 5

def test_divide_floats():
  """Test dividing floating-point numbers."""
  assert divide(5.0, 2.0) == 2.5

def test_divide_by_zero():
  """Test that dividing by zero raises ValueError."""
  with pytest.raises(ValueError):
    divide(10, 0)

def test_divide_type_error():
  """Test that divide raises TypeError for non-numeric input."""
  with pytest.raises(TypeError):
    divide("a", 3)
  with pytest.raises(TypeError):
    divide(5, "b")

def test_divide_zero_by_number():
  """Test dividing zero by a non-zero number."""
  assert divide(0, 5) == 0
