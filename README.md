NIPS 2003 workshop on feature extraction (datasets)

http://clopinet.com/isabelle/Projects/NIPS2003/



Challenge result analysis

http://clopinet.com/isabelle/Projects/NIPS2003/analysis.html



### Datasets

All of the datasets are two-class classification.



**DEXTER**

Domain: Text Classification

Data type: sparse-integer

Number of features: 20000

|           | Pos_ex | Neg_ex | Tot_ex | Check_sum   |
| --------- | ------ | ------ | ------ | ----------- |
| **Train** | 150    | 150    | 300    | 2816528.00  |
| **Valid** | 150    | 150    | 300    | 2820952.00  |
| **Test**  | 1000   | 1000   | 2000   | 19127295.00 |
| **All**   | 1300   | 1300   | 2600   | 24764775.00 |

**GISETTE**

Domain: Digit recognition

Data type: non-sparse

Number of features: 5000

|           | Pos_ex | Neg_ex | Tot_ex | Check_sum     |
| --------- | ------ | ------ | ------ | ------------- |
| **Train** | 3000   | 3000   | 6000   | 3164568508.00 |
| **Valid** | 500    | 500    | 1000   | 535016668.00  |
| **Test**  | 3250   | 3250   | 3250   | 3431572010.00 |
| **All**   | 6750   | 6750   | 13500  | 7131157186.00 |

**MADELON**

Domain: Artificial

Data type: non-sparse

Number of features: 500

|           | Pos_ex | Neg_ex | Tot_ex | Check_sum     |
| --------- | ------ | ------ | ------ | ------------- |
| **Train** | 1000   | 1000   | 2000   | 488083511.00  |
| **Valid** | 300    | 300    | 600    | 146395833.00  |
| **Test**  | 900    | 900    | 1800   | 439209553.00  |
| **All**   | 2200   | 2200   | 4400   | 1073688897.00 |

### Method

Feature Selection + PCA +  Classifier

Classifier: Markov Chain Monte Carlo, KNN

compare error rates using confusion matrix for different datasets



### Result

null