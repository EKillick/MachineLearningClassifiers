package machinelearningcw;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An implementation of a basic LinearPerceptron 
 * that implements the Weka Classifier class
 * @author 6277497
 */
public class LinearPerceptron extends AbstractClassifier{
    
    private int stoppingIterations; //stopping condition if no solution
    private double weights[];
    private double learningRate;
    private double bias;
    
    //Constructors
    
    /**
     * Default constructor for a LinearPerceptron Classifier
     */
    public LinearPerceptron(){
        this.learningRate = 1.0; // defaults to learning rate of 1.0
        this.stoppingIterations = 100; // default value of 100 iterations
        this.bias = 0.0;
    }
    
    /**
     * Constructor to allow setting of learning rate and stopping iterations 
     * @param l learningRate as a double
     * @param s stoppingIterations as an int
     * @param b optional bias
     */
    public LinearPerceptron(double l, int s, double b){
        this.learningRate = l;
        this.stoppingIterations = s;
        this.bias = b;
    }

    /**
     * Builds classifier from given Instance
     * @param instances of type Instances
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        int nonClassAttributes = (instances.numAttributes() - 1);
        weights = new double[nonClassAttributes]; // creates weights array
        Arrays.fill(weights, 1); // initialises weights to 1
        
        int iterationCount = 0;
        int correctCount = 0;
        
        //Check values are real
        for(int att = 0; att < instances.numAttributes()-1; att++){
            if(!instances.attribute(att).isNumeric()){
                throw new Exception("Non Real Values");
            }
        }
        
        do{
            for (int i = 0; (i < instances.numInstances() && correctCount 
                    < instances.numInstances()); i++){
                System.out.println(i);
                Instance instance = instances.get(i);
                double calculated = classifyInstance(instance);
                double actual = instance.classValue();
                double classDiff = actual - calculated;
                
                if (classDiff == 0.0){
                    correctCount++;
                } 
                else{
                        correctCount = 0;
                        //update the weights
                        for (int j = 0; j < nonClassAttributes; j++){
                        weights[j] += (learningRate * classDiff * instance.value(j));
                    }
                }
                iterationCount++;
            }
        }while(iterationCount < stoppingIterations && correctCount < instances.numInstances());
    }

    /**
     * Classifies a single instance
     * @param instnc instance to classify
     * @return 0 or 1
     * @throws Exception 
     */
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
       double result = 0;
       
        // multiply values and weights
        for (int i = 0; i < instnc.numAttributes() - 1; i++){
            result += (weights[i] * instnc.value(i));
        }
        result += bias;

        if (result > 0){
            return 1.0;
        }
        else{
            return 0.0;
        }
    }
}
