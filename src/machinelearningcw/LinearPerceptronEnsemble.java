package machinelearningcw;

/**
 * An ensemble of EnhancedLinearPerceptron classifiers
 * @author EKillick
 */
public class LinearPerceptronEnsemble {
    private int ensembleSize;
    private EnhancedLinearPerceptron[] arrayEnsemble;
    
    /**
     * Default constructor for a LinearPerceptronEnsemble
     */
    public LinearPerceptronEnsemble(){
        ensembleSize = 50;
        arrayEnsemble = new EnhancedLinearPerceptron[ensembleSize];
    }
}
