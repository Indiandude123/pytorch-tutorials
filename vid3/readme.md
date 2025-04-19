# Pytorch Autograd

The why?
Say we have a mathematical expression 
y = x^2. And we are given the challenge that given any value of x we can find the derivative
of y wrt x. So basically you would do dydx = 2x, just return this 

But what if now we have:
y = x^2 
z = sin(y)
and we need to find dzdx which we will calculate using chain rule of 
differentiation. 

Now what if 
y = x^2
z = sin(y)
u = exp(z)
then the complexity will increase even more. 

So as the nested function becomes more complex, writing the b
derivative and coding it up becomes more difficult. And deep 
learning involves nested function along with their derivatives.

Autograd is a great feature in pytorch which provides automatic differentiation for tensor operations. 