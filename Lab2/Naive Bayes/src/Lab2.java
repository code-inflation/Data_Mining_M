import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class Lab2 {
	public static void main(String[] args)
	{
		BufferedReader reader;
		Instances data = null;
		try {
			reader = new BufferedReader(new FileReader("dataset/breast-cancer.arff")); //Reading dataset
			Random rand = new Random(System.currentTimeMillis()); // Create seeded number generator
			data = new Instances(reader);   // Create Instances from read data: Instances is a class for handling an ordered set of weighted instances.
			data.randomize(rand);  // Randomize the position of instances.
			reader.close(); 
		} catch (Exception e) {
			e.printStackTrace();
		}		
		System.out.println(data);
		data.setClassIndex(data.numAttributes() - 1); //Setting the last attribute as class attribute
	    Instances train = data.trainCV(2, 0);  //Creating instances for train taking the half first part
	    Instances test = data.testCV(2, 1);    //Creating instances for test taking the half last part
	    try {
	    		NaiveBayes model=new NaiveBayes(); // Creating Naive Bayes model
			model.buildClassifier(train); //Building the classifier using the training Instances
			Evaluation eval_train = new Evaluation(test); // Creating the evaluation with test instances
		    eval_train.evaluateModel(model,test); //Doing the evaluation specifying which model and test instances use
		    System.out.println(eval_train.toMatrixString()); //Printing the confusion matrix
		    System.out.println(eval_train.toSummaryString());//Printing the percentage of model
		} catch (Exception e) {				
			e.printStackTrace();
		}   
	}		
}