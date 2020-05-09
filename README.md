# Utils
Some useful scripts for a ML Pipeline
<h3> Custom_Export_Graphviz </h3>

This script is an altered version of Sklearn's tree._export. It allows to plot a Decision Tree which has the original class 
distributions in the nodes, while using the class_weight parameter. 

When you use class_weight parameter in a Decision Tree, Sklearn shows values in the nodes as weighted. For example let's say that you use
class_weight as {0:1,1:5}. In your plotted tree, a node has a value of [100 100], and sample size as 120. This can be misleading 
because according to the value, there are 200 samples, but in reality this node has 120 samples, so the value should be [100 20].
This is the cause of weighted classes. 

With this script, you can call export_graphviz with an additional class_weight parameter and, granted that you entered your tree's original class_weight, it will return a representation of your tree with the actual class sizes in the nodes.

<h4>Usage:</h4>


```python
from Myazan_Utils import Custom_Export_Graphviz 

Custom_Export_Graphviz.export_graphviz(... (Standard parameters of export_graphviz)), class_weight = {class_weights of your tree} (default is "balanced"))
```

<h3> Feature_Class_Distribution </h3>

This script is influenced by WOE binning. WOE is used to determine a features predictive power. It seperates the values of
a feature into bins, and determines the class distributions in that bin. This script uses Decision Tree Classifier to determine the bins.

It takes a feature, creates a Decision Tree Classifier with only that feature, gets the thresholds of the nodes, 
and uses these thresholds to cut the feature into bins. This method can be utilized as a feature importance tool, and it also helps with the interpretability of the model. You can see which features are good at splitting your target value. 

Feature Class Distribution takes the dataframe and the name of your target value as parameters. It individually works for every 
feature in your dataframe. Depth and Min Leaf Node Size of the Tree's can also be given as parameters. It returns a dataframe where
all the features are seperated into bins. A row consists of a bin, it has those columns: Name of the Feature, 
Starting value of the Bin, Ending Value of the Bin, Count of Class 1, Count of Class 0, Total Count, Distribution of Class1.

Besides returning the results in a dataframe, this method also creates the graphs of all the trees created, it names each tree with the
feature it worked and puts it into a folder.

<h4> Usage </h4>

```python
from Myazan_Utils import Feature_Class_Distribution

Feature_Class_Distribution(df, target, max_depth = 3, min_leaf_node_size = 100)
```
