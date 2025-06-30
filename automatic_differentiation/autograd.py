# File: autograd.py
# Created: 20 April 2025
# -------------------------
# Summary
# -------------------------
# This file implements the Gradient class
# that allows reverse-mode autodifferentiation
# or backpropagation for scalars.


import graphviz


class Gradient():
    def __init__(self, value, _prev=(), _operation=" "):
        assert value is not None, "Need to enter a value"
        assert isinstance(value, (int, float)), "Value needs to be an integer or float"

        self.value = value 
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_prev)                      # What happens if set not used??
        self._operation = _operation


    def __add__(self, addendend):
        assert isinstance(addendend, Gradient),"The two numbers need to be wrapped in the Gradient class"
        sum = Gradient(self.value + addendend.value, (self, addendend), "+")

        def _backward():
            return 1

        sum._backward = _backward
        return sum
        

    def __mul__(self, factor):
        assert isinstance(addendend, Gradient), "The two numbers need to be wrapped in the Gradient class"
        product = Gradient(self.value * factor.value), (self, factor)

        def _backward():
            return factor

        product._backward = _backward
        return product
            

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "The class can only handle int/float exponents"
        result = Gradient(self.value**exponent, (self,), f"**{exponent}")

        def _backward():
            return (exponent * self.value ** (exponent - 1)) * 1
        result._backward = _backward
        return result


    def visualise_graph(self):
        graph = graphviz.Digraph()
        visited = set()

        def traverse(node):
            if not node or node in visited:
                return

            visited.add(node)
            node_id = str(id(node))
            label = f"Value={node.value:.2f} | Grad={node.grad:.2f} | OP={node._operation}"
            graph.node(node_id, label=label, shape="box")

            for child in node._prev:
                child_id = str(id(child))
                graph.edge(child_id, node_id)
                traverse(child)

        traverse(self) 
        return graph

    def _topolocial_sort():
        pass


    def backward(self):
        pass 
    
