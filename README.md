# ML-Python-scikit-learn
Here is the implementation of several Machine Learning algorithms in Python.

This python code is part of the research, whose motivation and results are descripted in the paper: 
***Predicting User Reactions to Twitter Feed Content based on Personality
Type and Social Cues***   https://doi.org/10.1016/j.future.2019.10.044

Here are some concepts that are useful for a correct interpretation of some code details:
- *Dataset*: The Twitter dataset that we used for our experimental evaluation contains 18,292,721 posts published between July 15, 2013 and March 25, 2015; originally collected as part of an effort to analyze elections in India. 
- *Spread*: A typical characteristic of real-world social network datasets is that there is great variability in user behavior—there are many users who most very little (nothing, or next to nothing), as well as many who are very active in their posting habits. In order to investigate the effect that this class imbalance has on the performance of our classifiers, we selected users according to a parameter that we call spread; a user will be chosen according to a value of x for this parameter if the difference (in percentage points) between the percentage of intervals for which action was taken vs. no action was taken is at most x. So, for instance, for a value of 60, a user with 23% intervals with action and 77% is selected (since 77 − 23 = 54 ≤ 60, but one with 19%–81% is not (since 81 − 19 = 62 > 60).
- *OCEAN*: The personality type as a feature was provided by the Personality Insights service by IBM Cloud (https://www.ibm.com/watson/services/personality-insights/). The values were discretized into high and low (or "+" and "–") to obtain a value between 1 and 32. 
