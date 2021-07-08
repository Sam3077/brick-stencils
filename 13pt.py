from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)

# Express computation
# output[i, j, k] is assumed
calc = input(i, j, k) + \
       input(i + 1, j, k) + input(i + 2, j, k) + \
       input(i - 1, j, k) + input(i - 2, j, k) + \
       input(i, j + 1, k) + input(i, j + 2, k) + \
       input(i, j - 1, k) + input(i, j - 2, k) + \
       input(i, j, k + 1) + input(i, j, k + 2) + \
       input(i, j, k - 1) + input(i, j, k - 2)
output(i, j, k).assign(calc)

STENCIL = [output]
