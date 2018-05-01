package machinelearningcw;

import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * An enhanced perceptron based on the LinearPerceptron
 * that implements the Weka Classifier class
 * @author 6277497
 */
public class EnchancedLinearPerceptron extends AbstractClassifier{
    
    private int stoppingIterations;
    private double weights[];
    private double learningRate;
    
    //Constructors
    
    /**
     * Default constructor for a LinearPerceptron Classifier
     */
    public EnchancedLinearPerceptron(){
        this.learningRate = 1.0; // defaults to learning rate of 1.0
        this.stoppingIterations = 100; // default value of 100 iterations
    }
    
    /**
     * Constructor to allow setting of learning rate and stopping iterations 
     * @param l learningRate as a double
     * @param s stoppingIterations as an int
     */
    public EnchancedLinearPerceptron(double l, int s){
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
     * @param instances
     * @param i of type Instance
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        weights = new double[instances.numAttributes() - 1]; // creates weights array
        Arrays.fill(weights, 1); // initialises weights to 1
        
        int iterationCount = 0;
        int correctCount = 0;
        
        do{
            for (int i = 0; i < instances.numInstances(); i++){
                Instance instance = instances.get(i);
                double calculated = classifyInstance(instance);
                double actual = instance.classValue();
                double classDiff = actual - calculated;
                
                if (classDiff == 0.0){
                    correctCount++;
                } 
                else{
                    correctCount = 0;
                }

                //update the weights
                for (int j = 0; j < instances.numAttributes() - 1; j++){
                    weights[j] += (learningRate * classDiff * instance.value(j));
                }
                System.out.println("\n");
                
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
       
       if (result > 0){
           return 1.0;
       }
       else{
           return 0.0;
       }
    }
}
