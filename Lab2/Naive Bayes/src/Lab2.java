import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class Lab2 {
	public static void main(String[] args)
	{
		BufferedReader reader;
		Instances data = null;
		try {
			reader = new BufferedReader(new FileReader("Naive Bayes/dataset/breast-cancer.arff")); //Reading dataset
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
	    Evaluation eval = null; 
	    try {
	    		NaiveBayes model=new NaiveBayes(); // Creating Naive Bayes model
			model.buildClassifier(train); //Building the classifier using the training Instances
			eval= new Evaluation(test); // Creating the evaluation with test instances
		    eval.evaluateModel(model,test); //Doing the evaluation specifying which model and test instances use
		    System.out.println(eval.toMatrixString()); //Printing the confusion matrix
		    System.out.println(eval.toSummaryString());//Printing the percentage of model
		} catch (Exception e) {				
			e.printStackTrace();
		}   
	    // generate curve
	     ThresholdCurve tc = new ThresholdCurve();
	     int classIndex = 0;
	     Instances result = tc.getCurve(eval.predictions(), classIndex);
	 
	     // plot curve
	     ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
	     vmc.setROCString("(Area under ROC = " + Utils.doubleToString(tc.getROCArea(result), 4) + ")");
	     vmc.setName(result.relationName());
	     PlotData2D tempd = new PlotData2D(result);
	     tempd.setPlotName(result.relationName());
	     tempd.addInstanceNumberAttribute();
	     // specify which points are connected
	     boolean[] cp = new boolean[result.numInstances()];
	     for (int n = 1; n < cp.length; n++)
	       cp[n] = true;
	     try {
			tempd.setConnectPoints(cp);
			// add plot
		     vmc.addPlot(tempd);
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	     
	 
	     // display curve
	     String plotName = vmc.getName();
	     final javax.swing.JFrame jf =
	       new javax.swing.JFrame("Weka Classifier Visualize: "+plotName);
	     jf.setSize(500,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     jf.getContentPane().add(vmc, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	       jf.dispose();
	       }
	     });
	     jf.setVisible(true);
	}		
}