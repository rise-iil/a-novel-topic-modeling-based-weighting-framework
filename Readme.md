# A novel topic modeling based weighting framework for class imbalance learning.

## List of Files
* confusionmatStats.m -> Function which computes the Precision, Recall, F-score and Confusion Matrix 
* CSVtoARFF.m -> Function which converts a CSV file to ARFF file
* plsa2.m -> Function which implements the PLSA algorithm
* run_PLSA.m -> Wrapper Function to call PLSA
* test_todus.m -> Wrapper Function to perform 5-fold CV TODUS Algorithm
* TODUS.m -> Function which implements the TODUS Algorithm

## Usage

TODUS Algorithm and its variants are used for efficient learning from unbalanced datasets. TODUS, uses the following WEKA classfiers to learn from the re-sampled data,</br>
	1. SVM</br>
	2. DecisionTreeClassifier(J48)</br>
	3. kNN</br>
	4. Logistic</br>

### Input Format

- The input to all the variants of the algorithm, is a CSV file, where each line is of the format:</br>
		[attr_1_value],[attr_2_value],......,[attr_n_value],[label]</br>
		Here, *label* is either 1 or -1</br>
    1 -> Minority Class</br>
	  -1 -> Majority Class
- An ARFF header file corresponding to the CSV file
		You can refer the format for ARFF header file in [here](http://www.cs.waikato.ac.nz/ml/weka/arff.html)
		
### Running the code for TODUS

* First, specify the location of the CSV file in the wrapper function of the Algorithm to be run</br>
		file = '**location of the CSV file**'; 
* Specify the location of the ARFF header in line 14 of CSVtoARFF.m file</br>
		fid = fopen('**location of the ARFF Header file**','r');
* Run the wrapper function

		
### Output Format

The output of all the wrapper functions is a k x 5 struct matrix variable named 'output', where k is the number of folds.

Each 1 x 5 vector of output represents,
* 1st Column -> Confusion Matrix in the form {True positives, True Negatives, False Positives, False Negatives}
* 2nd Column -> Precision values in the form {Precision of Majority class, Precision of Minority Class}
* 3rd Column -> Recall values in the form {Recall of Majority class, Recall of Minority Class}
* 4th Column -> F-score values in the form {F-score of Majority class, F-score of Minority Class}
* 5th Column -> Group Order

The Area Under the Curve(AUC) values are stored in the auc array.
