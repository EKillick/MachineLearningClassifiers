package machinelearningcw;

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
        this.learningRate = 1.0; // defaults to learning rate of 0
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

    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
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
