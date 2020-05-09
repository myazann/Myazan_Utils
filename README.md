# Utils
Some useful scripts for a ML Pipeline
<h3> Custom_Export_Graphviz </h3>

This script is an altered version of Sklearn's tree._export. It allows to plot a Decision Tree which has the original class 
distributions in the nodes, while using the class_weight parameter. 

When you use class_weight parameter in a Decision Tree, Sklearn shows values in the nodes as weighted. For example let's say that you use
class_weight as {0:1,1:5}. In your plotted tree, a node has a value of [100 100], and sample size as 120. This can be misleading 
because according to the value, there are 200 samples, but in reality this node has 120 samples, so the value should be [100 20].
This is the cause of weighted classes. 

With this script, you can call export_graphviz with an additional class_weight parameter and, granted that you entered your tree's original 
class_weight, it will return a representation of your tree with the actual class sizes in the nodes.

<h4>Usage:</h4>

import Custom_Export_Graphviz 

Custom_Export_Graphviz.export_graphviz(... (Standard parameters of export_graphviz)),class_weight = {class_weights of your tree} (default is "balanced"))
