package machinelearningcw;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;

/**
 * An enhanced perceptron based on the LinearPerceptron
 * that implements the Weka Classifier class
 * @author 6277497
 */
public class EnhancedLinearPerceptron extends AbstractClassifier{
    
    private int stoppingIterations;
    private double weights[];
    private double learningRate;
    private double bias;
    private boolean standardisation;
    private boolean modelSelection;
    private double[] attributeMeans;
    private double[] attributeStdDev;
    private boolean useOnline;
    
    //Constructors
    
    /**
     * Default constructor for a LinearPerceptron Classifier
     */
    public EnhancedLinearPerceptron(){
        this.learningRate = 1; // defaults to learning rate of 1.0
        this.stoppingIterations = 100; // default value of 100 iterations
        this.standardisation = true; 
        this.modelSelection = false;
        this.bias = 0.0;
        this.useOnline = true;
    }
    
    /**
     * Constructor with default values for everything but standardisation and
     * model selection
     * @param std
     * @param ms 
     */
    public EnhancedLinearPerceptron(boolean std, boolean ms){
        this(1, 100, std, ms, 0, true);
    }
    
    /**
     * Constructor to allow setting of learning rate and stopping iterations 
     * @param l learningRate as a double
     * @param s stoppingIterations as an int
     * @param std defines whether or not to standardise
     * @param ms defines whether or not to use model selection
     * @param bias bias that is added before classification
     */
    public EnhancedLinearPerceptron(double l, int s, boolean std, boolean ms,
            double bias, boolean useOnline){
        this.learningRate = l;
        this.stoppingIterations = s;
        this.standardisation = std;
        this.modelSelection = ms;
        this.bias = bias;
        this.useOnline = useOnline;
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
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        int nonClassAttributes = (instances.numAttributes() - 1);
        attributeMeans = new double[nonClassAttributes];
        attributeStdDev = new double[nonClassAttributes];
        
        weights = new double[nonClassAttributes]; // creates weights array
        Arrays.fill(weights, 1); // initialises weights to 1
        
        if (this.modelSelection){
            Evaluation evalOffline = new Evaluation(instances);
            EnhancedLinearPerceptron offClassifier = new EnhancedLinearPerceptron();
            offClassifier.useOnline = false;
            offClassifier.buildClassifier(instances);
            evalOffline.crossValidateModel(offClassifier, instances, 5, new Random());
            
            Evaluation evalOnline = new Evaluation(instances);
            EnhancedLinearPerceptron onClassifier = new EnhancedLinearPerceptron();
            onClassifier.buildClassifier(instances);
            evalOnline.crossValidateModel(onClassifier, instances, 5, new Random());    
            
//            System.out.println("Offline Error: " + evalOffline.errorRate());
//            System.out.println("Online Error: " + evalOnline.errorRate());
            
            if (evalOffline.errorRate() < evalOnline.errorRate()){
                useOnline = false;
            }
        }
        
        if (useOnline){
            this.buildOnlineClassifier(instances);
        }
        else{
            this.buildOfflineClassifer(instances);
        }
        
    }
    
    /**
     * Builds an online classifier on a given instance
     * @param instances
     * @throws java.lang.Exception
     */
    public void buildOnlineClassifier(Instances instances) throws Exception{
        int nonClassAttributes = (instances.numAttributes() - 1);
        
        for(int att = 0; att < nonClassAttributes; att++){
            //Check values are real
            if(!instances.attribute(att).isNumeric()){
                throw new Exception("Non Real Values");
            }
            AttributeStats attStats = instances.attributeStats(att);
            Stats stats = attStats.numericStats;
            attributeMeans[att] = stats.mean;
            attributeStdDev[att] = stats.stdDev;
        }

        int iterationCount = 0;
        int correctCount = 0;
        double value;
        
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
                for (int j = 0; j < nonClassAttributes; j++){
                    if (standardisation){
                    value = standardiseValue(instance.value(j), 
                                        attributeMeans[j], attributeStdDev[j]);
                    }
                    else{
                        value = instance.value(j);
                    }
                    weights[j] += (learningRate * classDiff * value);
                }
                
                iterationCount++;
            }
        }while(iterationCount < stoppingIterations && correctCount < instances.numInstances());
    }
    
    /**
     * Builds an offline classifier on a given instance
     * @param instances 
     * @throws java.lang.Exception 
     */
    public void buildOfflineClassifer(Instances instances) throws Exception{
        int nonClassAttributes = (instances.numAttributes() - 1);
        
        double[] deltaWeights;
        deltaWeights = new double[nonClassAttributes];
        
        for(int att = 0; att < nonClassAttributes; att++){
            //Check values are real
            if(!instances.attribute(att).isNumeric()){
                throw new Exception("Non Real Values");
            }
            AttributeStats attStats = instances.attributeStats(att);
            Stats stats = attStats.numericStats;
            attributeMeans[att] = stats.mean;
            attributeStdDev[att] = stats.stdDev;
        }

        int iterationCount = 0;
        int correctCount = 0;
        double value;
        
        do{
            Arrays.fill(deltaWeights, 0);
            for (int i = 0; i < instances.numInstances(); i++){
                Instance instance = instances.get(i);
                double calculated = classifyInstance(instance);
                double actual = instance.classValue();
                double classDiff = actual - calculated;
//                System.out.println(calculated);
                
                if (classDiff == 0.0){
                    correctCount++;
                } 
                else{
                    correctCount = 0;
                }
                
                //update the weights
                for (int j = 0; j < nonClassAttributes; j++){
                    if (standardisation){
                    value = standardiseValue(instance.value(j), 
                                        attributeMeans[j], attributeStdDev[j]);
                    }
                    else{
                        value = instance.value(j);
                    }
                    deltaWeights[j] += (learningRate * classDiff * value);
                }
            }
            for (int k = 0; k < nonClassAttributes; k++){
                weights[k] += deltaWeights[k];
            }
//            System.out.println(weights[0] + " " + weights[1]);
            iterationCount++;
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
        double value;
       
         // multiply values and weights
        for (int i = 0; i < instnc.numAttributes() - 1; i++){
            if (standardisation){
                value = standardiseValue(instnc.value(i), 
                                     attributeMeans[i], attributeStdDev[i]);
                }
                else{
                    value = instnc.value(i);
            }
            result += (weights[i] * value);
       }
       result += bias;
       
       if (result > 0){
           return 1.0;
       }
       else{
           return 0.0;
       }
    }
    
    /**
     * A method to standardise values
     * @param value
     * @param mean
     * @param stdDev
     * @return 
     */
    private double standardiseValue(double value, double mean, double stdDev){
        return ((value - mean)/stdDev);
    }
}
