|    | Experiment           |   Accuracy | Confusion Matrix   | Comment                                                                       |
|---:|:---------------------|-----------:|:-------------------|:------------------------------------------------------------------------------|
|  0 | logistic_regression  |   0.8125   | [[119  11]         | My Logistic Regression solution and the labels are pregnant, glucose, bp, bmi |
|    |                      |            |  [ 25  37]]        |                                                                               |
|  1 | SGDClssifier         |   0.744792 | [[108  22]         | solution with SGDC and the labels are pregnant, glucose, bp, bmi              |
|    |                      |            |  [ 27  35]]        |                                                                               |
|  2 | DecisionTreeClassier |   0.755208 | [[119  11]         | solution with DTC and the labels are pregnant, glucose, bp, bmi               |
|    |                      |            |  [ 36  26]]        |                                                                               |
|  3 | KNeighborClassofier  |   0.770833 | [[118  12]         | solution with KNC and the labels are pregnant, glucose, bp, bmi               |
|    |                      |            |  [ 32  30]]        |                                                                               |
