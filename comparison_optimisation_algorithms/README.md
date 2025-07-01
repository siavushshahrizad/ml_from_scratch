# Introduction
This project explores the performance of different optimisation algorithms. I  compared these algorithms on a dataset for children's well-being and looked at how these algorithms if we use only one predictor and then several. 

# Algorithms Compared
- **Closed-form solution**: Direct matrix computation
- **Gradient descent**: First-order iterative method  
- **Newton's method**: Second-order optimization using Hessian
- **BFGS**: Quasi-Newton method approximating second-order information

# Usage
If you want to run any of the code, you need to create a venv and install requirements.

# Dataset and procedure
The project uses the 2008 [National Survey of Young People's Well-Being](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=7898#!/documentation). The data comes from a sample of 10,000 children 10 to 15 years old, collected throughout England on behalf of the Children's Society. Note though that there only 6000ish observations in the actual dataset.

I looked at an overall measure of well-being as the outcome, measured from 1-10 with 10 defined as the "Best possible life" (Q3). In the one-variable case of my investigation, I used strength of family relationships as the predictor, also measured from 1-10 with higher scores meaning more satisfaction with the family.

# Algorithm lessons

# Thoughts on the domain
