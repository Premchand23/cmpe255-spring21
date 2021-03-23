| Experiement               | Accuracy | Confusion Matrix     | Comment                                                                 |
|---------------------------|----------|----------------------|-------------------------------------------------------------------------|
| Baseline                  | 0.677083 | [[114  16] [46  16]] | Base solution                                                           |
| logistic_regression       | 0.8215   | [[119  11] [25  37]] | solution with LGR and the labels are 'pregnant', 'glicose', 'bp', 'bmi' |
| SGDClassifier             | 0.744792 | [[108  22] [27  35]] | solution with SGD and the labels are 'pregnant', 'glicose', 'bp', 'bmi' |
| DecisionTreeClassifier    | 0.770833 | [[119  11] [36  26]] | solution with DTC and the labels are 'pregnant', 'glicose', 'bp', 'bmi' |
| KNeighborClassifier       | 0.770833 | [[118  12] [32  30]] | solution with KNC and the labels are 'pregnant', 'glicose', 'bp', 'bmi' |
