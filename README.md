"ARF" = Auton Random Forest, the C library

arfreader.py : Converts an ARF file to an sklearn RF. Only classifiers are implemented right now. Adapated from Nick's code

arfsk.py : Wraps an sklearn RF instance to provide the metrics in the C library: inbounds, dot product sum, mean entropy, feature usages

Contact Dan Howarth with any questions


## Tree prediction metrics

# Dot product sum

Consider a forest with N trees. There are $k = { N \choose 2}$ pairs. Given data point $x$, Let $p_i$ be the column vector of class probabilities for the ith tree. The dot product sum is 
the sum of dot products over these $k$ pairs:

$$dps(x) = \sum_{i=1}^N \sum_{j > i}^N p_i^T p_j$$

## Write out to AutonRF format

```
import arfsk
wrapper = arfsk.AutonSKRFWrapper(rf) # rf is the sklearn random forest
wrapper.write_legacy("desired/output/filename.txt", ignore_inbounds=True)
```

If you want to include the inbounds information in a way that matches AutonRF you have to first use the train data to set the bounds.

```
import arfsk
wrapper = arfsk.AutonSKRFWrapper(rf) # rf is the sklearn random forest
wrapper.set_bounds_data(train_data)
wrapper.write_legacy("desired/output/filename.txt", ignore_inbounds=False)
```
