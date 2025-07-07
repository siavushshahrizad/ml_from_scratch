# Introduction
I compared two different optimisation algorithms for linear regression. 
- **Closed-form solution**: An algorithm that provides us instantly with the best-possible weights
- **Gradient descent**: An algorithm that improves weights iteratively

I was interested in how they performed against each other and when one might be more feasible then the other.

# Background
Gradient descent is the dominant optimisation algorithm in practice. The closed-form algorithm is an alternative, which textbooks tell us doesn't scale well and isn't available for non-linear problems such as logistic regression and neural networks. I have added technical details on these issues, such as derivation of gradients, [here](./linear_regression.pdf). One point, for example, is that the closed-form algorithm requires O(D**2) space with D being the number of features, whereas gradient descent only needs O(D), meaning the latter scales better. One reason we would want to use closed-form is because it gives us for sure the best weights, whereas theoretically we could stop gradient descent too soon and end up with less-than-optimal weights.

# Data
I used the California Housing dataset,  which contains about 20k samples to predict house prices based on 8 variables such as median income. More details can be found [here](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).

# Lessons
1. Gradient descent generally achieves better validation losses, though with maybe 500-1000 epochs of weight updates, Gradient descent might get there. I just did a flat implementation where there were 100 epochs for all simulated scenarios.


# Meta-lessons
1. **Data is hard**. The closed-form algorithm is supposed to break down, for example, when there are too many features in the dataset, but finding such a dataset is hard. Datasets aren't that well annotated often, and many contain few features. So I worked with suboptimal data, and augmented it with randomly initiated, synthetic features. A better dataset would probably be one in the language domain or genetic data. Finding semi-suitable data was a much longer time-sink than I thought.
2. **Formulas  oversimplify**. Textbooks will tell you what the formula for the forward pass might be, and you feel clever when you implement it. What makes you feel dumb, on the other hand is, when your losses look weird and you spend hours and hours looking it weights and their updates. I learned that one solution to my problems was data normalisation. Textbooks rarely tell you that correct implementation of a formula does not equal a functioning algorithm. A successful ML programme requires a lot more debugging, archaeology, and a paranoia to check your work than I thought. I guess my expectations got a weight update!
3. **Believe the smoking gun**. I think a lot of engineering success is about psychology. We want our algorithms to work, so there is confirmation bias. I noticed initially that my algorithm seemed to work in many places, and I had to force myself to look into those because it was just so seductive to take the chips that were already on the table. But my losses seemed weird, there was an explosion of the loss when their was one epoch of learning bu then essentially massive learning followed by immediate stagnation in the second epoch. This was due to several reasons such as overflow and the variety of scales on which the Housing data was measured on.





