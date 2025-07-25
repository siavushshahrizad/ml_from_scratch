# File: test_autograd.py
# Created: 20 April 2025
# -------------------------
# Summary
# -------------------------
# This file implements unit tests for the  Gradient class.
# Test results are validated against the PyTorch library.


import torch
import pytest
from autograd import Gradient


class Test_Gradient():
    def test_simple_visualisation(self):        # Largely a dummy test; need to visually inspect the created file
        x = Gradient(5)
        z = x**2 
        graph = z.visualise_graph()
        graph.render("simple_test_visualistion", format="png")
        assert True


    @pytest.mark.skip(reason="add and mul needed before testing")
    def test_complex_visualisation(self):       # Ditto
        x = Gradient(5)
        assert True


    @pytest.mark.skip(reason="backward not ready")
    def test_simple_topological_sort(self):
        pass


    @pytest.mark.skip(reason="backward not ready")
    def test_complex_topological_sort(self):
        pass


    @pytest.mark.skip(reason="backward not ready")
    def test_simple_gradient_with_addition(self):
        pass

    
    @pytest.mark.skip(reason="backward not ready")
    def test_simple_gradient_with_multiplication(self):
        pass


    @pytest.mark.skip(reason="backward not ready")
    def test_simple_gradient_with_exponent(self):
        x = Gradient(5)
            
        y = x ** 2
        y.backward()

        x_tensor = torch.tensor(5.0, requires_grad=True)
        y_tensor = x_tensor ** 2
        y_tensor.backward()
        assert int(x_tensor.grad.item()) == x.grad
