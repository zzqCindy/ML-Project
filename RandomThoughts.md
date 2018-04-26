### Feature Selection

- Mutual Information
  - non-binary data compared to the median
  - zero valued featured were assumed to be sleeping
- Dist Based
  - defined proper margin for prototype based algorithms(KNN, SVM)
  - margin is the difference between the distance to the closest negative to closest positive
  - select the feature that maximizes this margin
- Correlation Ranking
  - $ CR_j = \frac{|(x_j-\mu_j)^T(y-\mu_y)|}{|x_j||y|}, j = 1,2,...,N_{Feat}$
  - $CR_j$ is rank of feature j, $x_j$ is feature vector j, y is class label vector, $\mu_j$ and $\mu_y$ are expectation values of feature j and class vector y respectively. $N_{Feat}$ is dimensionality of feature space.
- FDR Ranking
  - $FDR_j = \frac{(\mu_{j,1}-\mu_{j,2})^2}{\sigma^2_{j,1}+\sigma^2_{j,2}}, j = 1,2,...,N_{Feat}$
  - $FDR_j$ is rank of feature j, $\mu_{j,1}$ and $\mu_{j,2}$ are class mean value of feature vector j for class 1 and 2, respectively, $\sigma^2_{j,1}$ and $\sigma^2_{j,2}$ are class variance value of feature vector j for class 1 and 2, respectively, and $N_{Feat}$ is dimensionality of feature space
- Random Forests?





### Classifier

Naive Bayes, Perceptron Criterion (Gradient Descent), KNN, RLSC.



### Simple Statistical Measures

F scores, ROC.



### DEXTER

- Attempt1
  - Normalization
  - Feature Selection: Mutual Inofrmation
  - Classification: SVM
- Attempt2
  - Feature Selection: correlation criteria(about 5%)





### GISETTE

- Attempt1	
  - Normalization (maximum absolute value was set to 1)
  - Feature Selection: Mutual Information
  - Classification: aggressive perception with a limit set to 600
- Attempt2
  - Feature Selection: Features ranked using Fisher's discriminant criteria, those with higher values were selected (about 10%)
  - Preprocessing: Normalized and applied a linear PCA, those with low contribution to overall variance were removed





### MADELON

- Attempt1
  - Normalization (maximum absolute value was set to 1)
  - Feature Selection: Dist Based
  - Classfication: SVM
- Attempt2
  - Feature Selection: Fisher's discriminant criteria (about 2%)
  - Normalization





