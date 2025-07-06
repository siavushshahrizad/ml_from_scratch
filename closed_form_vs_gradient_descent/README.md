# Introduction
This project is about what doesn't work. Originally, I had set out to compare different optimisation algorithms, by implementing them from scratch and focusing on the linear regression problem. In the end, I compared these two:
- **Closed-form solution**: An algorithm that provides us instantly with the best-possible weights
- **Gradient descent**: An algorithm that adjust weights iteratively to improve them

Gradient descent is the dominant optimisation algorithm used in practice, and the staple of libraries such as HuggingFace's transformers library. The closed-form algorithm is an alternative, which textbooks tell us doesn't scale well and isn't available for non-linear problems such as logistic regression and neural networks. I have added technical detail on these issues, such as derivation of gradients, [here](./linear_regression.pdf). One point, for example, is that the closed-form algorithm requires O(D**2) space with D being the number of features, whereas gradient descent only needs O(D), meaning the latter scales better. Although modern GPUs should be be able to handle such space requirements up to a point.

One reason we would want to use closed-form is because it gives us for sure the best weights, whereas theoretically we could stop gradient descent too soon and end up with less-than-optimal weights. Although I originally wanted to show when the closed-form solution breaks down and why we would have to use gradient descent in practice, I found it hard to show this.

# Instructions
If you want to run any of the code, you need to create a venv and install requirements.

# Methods
I used the California Housing dataset,  which contains about 20k samples to predict house prices based on 8 variables such as median income. More details can be found [here](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

# Lessons
1. **Data availability**. It was hard to find the data I needed, and I spent a significant time searching and swapping datasets mid-stream. At some point tried to create randomly-initiated, synthetic features, for the California Housing dataset, but this impacted how much realistically gradient descent could learn, and I abandoned the approach. I really needed data with high numbers of features to show break-down, and if I had spent more time I could have probably found a dataset with many gene expression or a dataset in the language domain. In the end, my data was not right for the task, and this created many downstream problems.

2. **Silent failure**. Initially, when I still had synthetic data, I simulated matrix calculations with tens of thousands of features. On a first look, it appeared that the data was telling the story I wanted it to. But there were too many smoking guns; a lot of things looked weird. For example, the simulations never broke down even when I incr

2. **Measurement tools**. I measured the time and memory requirements of both algorithms with the time and tracemalloc library. I didn't spend enough time to fully understan


1. **Theory-data gap**: 1. **Data Availability**: Despite widespread ML adoption, finding datasets 
   that naturally demonstrate algorithmic limits proved surprisingly difficult.
2. **Implementation vs Theory**: NumPy's optimized implementations obscure 
   the theoretical O(n³) complexity through clever optimizations.
3. **Measurement Precision**: With small datasets, timing and memory 
   measurements are dominated by overhead and noise.

