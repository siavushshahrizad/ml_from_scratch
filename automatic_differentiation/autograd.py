# File: autograd.py
# Created: 20 April 2025
# -------------------------
# Summary
# -------------------------
# This file implements the Gradient class
# that allows reverse-mode autodifferentiation
# or backpropagation for scalars.

# TODO: Add Graphviz visualstion of the computational graph


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


    def __pow__(self, exponent):
        result = Gradient(self.value**exponent, (self,), f"**{exponent}")

        def _backward():
            pass
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
    
