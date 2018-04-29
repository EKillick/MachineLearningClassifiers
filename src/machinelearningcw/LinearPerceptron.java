package machinelearningcw;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An implementation of a basic LinearPerceptron 
 * that implements the Weka Classifier class
 * @author 6277497
 */
public class LinearPerceptron implements Classifier{
    
    private int stoppingIterations;
    private double weights[];
    private double learningRate;
    
    //Constructors
    
    /**
     * Default constructor for a LinearPerceptron Classifier
     */
    public LinearPerceptron(){
        this.learningRate = 1.0; // defaults to learning rate of 1.0
        this.stoppingIterations = 100; // default value of 100 iterations
    }
    
    /**
     * Constructor to allow setting of learning rate and stopping iterations 
     * @param l learningRate as a double
     * @param s stoppingIterations as an int
     */
    public LinearPerceptron(double l, int s){
        this.learningRate = l;
        this.stoppingIterations = s;
    }
    
    //Accessor Methods
    /**
     * Sets the learning rate
     * @param l - learning rate as a double
     */
    public void setLearningRate(double l){
        this.learningRate = l;
    }
    
    /**
     * Returns the learning rate
     * @return learning rate as a double
     */
    public double getLearningRate(){
        return this.learningRate;
    }
    
    /**
     * Sets the number of iterations before stopping
     * @param s - stopping conditions as an int
     */
    public void setStoppingIterations(int s){
        this.stoppingIterations = s;
    }
    
    /**
     * Returns the number of iterations before stopping
     * @return stopping iterations as an int
     */
    public int getStoppingIterations(){
        return this.stoppingIterations;
    }

    /**
     * Builds classifier from given Instance
     * @param i of type Instance
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances i) throws Exception {
        weights = new double[i.numAttributes() - 1]; // creates weights array
        Arrays.fill(weights, 1); // initialises weights to 1
        System.out.println(i.numAttributes());
        
        //
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
           result = (weights[i] * instnc.value(i));
       }
       
       if (result > 0){
           return 1;
       }
       else{
           return 0;
       }
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
