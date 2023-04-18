# RF-GAP-Python
Python code for Random Forest-Geometry- and Accuracy-Preserving Proximities

This package can be used to generate three types of random forest proximities which are described in the paper “Random Forest- Geometry- and Accuracy-Preserving Proximities” (https://ieeexplore.ieee.org/document/10089875). Currently, one can construct the Original, OOB, and RF-GAP proximiteis. A R-language version of this code is available at https://github.com/jakerhodes/RF-GAP-R.

# Useage:

The rfgap class is conditionally built upon the Sklearn RandomForestClassifier or RandomForestRegressor with the additional functionality of proximity generation via the method get_proximities. The factory function RFGAP requires the user to either provide a vector of labels, y, or to determine the prediction_type ("classification" or "regression"). If y is provided, the type of forest to by built will be determined automatically. In addition to any arguements used in RandomForestClassifier or RandomForestRegressor, an rfgap class takes the arguements prox_method (default: "rfgap") which, determines the proximity type to be constructed: "rfgap", "original", or "oob". The option to generate sparse or dense proximities is determined by the arguement matrix_type. See an example below:


```python 
prediction_type = 'classification'

rf = rfgap(prediction_type = prediction_type)
rf.fit(x, y)

proximities = rf.get_proximities()

```


This repository is still under construction. Additional features will be added to run proximity-based applications. 

