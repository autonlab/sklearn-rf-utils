"ARF" = Auton Random Forest, the C library

arfreader.py : Converts an ARF file to an sklearn RF. Only classifiers are implemented right now. Adapated from Nick's code

arfsk.py : Wraps an sklearn RF instance to provide the metrics in the C library: inbounds, dot product sum, mean entropy, feature usages

Contact Dan Howarth with any questions


## Tree prediction metrics

# Dot product sum

Consider a forest with N trees. There are $k = { N \choose 2}$ pairs. Given data point $x$, Let $p_i$ be the vector of class probabilities for the ith tree. The dot product sum is 
the sum of dot products over these $k$ pairs:

$$dps(x) = \sum_i=1^k \sum_j\neq k^k p_i^T p_i$$
