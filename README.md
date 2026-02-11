# RF-GAP-Python
Python code for Random Forest-Geometry- and Accuracy-Preserving Proximities

This package can be used to generate two types of random forest proximities which are described in the papers “Random Forest- Geometry- and Accuracy-Preserving Proximities” (https://ieeexplore.ieee.org/document/10089875) and "Scalable Tree Ensemble Proximities in Python" (https://arxiv.org/abs/2601.02735). Currently, one can construct the Original and RF-GAP proximities. An R-language version of this code is available at https://github.com/jakerhodes/RF-GAP-R.

# Installation
To install, please use ```pip install git+https://github.com/jakerhodes/RF-GAP-Python```

# Usage

The rfgap class is conditionally built upon the Sklearn RandomForestClassifier or RandomForestRegressor with the additional functionality of proximity generation via the method get_proximities. The factory function RFGAP requires the user to either provide a vector of labels, y or to determine the prediction_type ("classification" or "regression"). If y is provided, the type of forest to be built will be determined automatically. In addition to any arguments used in RandomForestClassifier or RandomForestRegressor, an rfgap class takes the arguments prox_method (default: "rfgap") which, determines the proximity type to be constructed: "rfgap", "oob" or "original". The option to generate sparse or dense proximities is determined by the argument matrix_type. See an example below:


```python
from rfgap import RFGAP
prediction_type = 'classification'

rf = RFGAP(prediction_type = prediction_type)
rf.fit(x, y)

proximities = rf.get_proximities()
```


This repository is still under construction. Additional features will be added to run proximity-based applications. 

# Citation

If you use this software in your research or experiments, please cite the following works:

```bibtex
@ARTICLE{10089875,
  author={Rhodes, Jake S. and Cutler, Adele and Moon, Kevin R.},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Geometry- and Accuracy-Preserving Random Forest Proximities}, 
  year={2023},
  volume={45},
  number={9},
  pages={10947-10959},
  keywords={Random forests;Forestry;Geometry;Data visualization;Decision trees;Task analysis;Anomaly detection;Proximities;random forests;supervised learning},
  doi={10.1109/TPAMI.2023.3263774}}
```

```bibtex
@misc{aumon2026scalabletreeensembleproximities,
      title={Scalable Tree Ensemble Proximities in Python}, 
      author={Adrien Aumon and Guy Wolf and Kevin R. Moon and Jake S. Rhodes},
      year={2026},
      eprint={2601.02735},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.02735}}
```


