package machinelearningcw;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import weka.filters.Filter;

/**
 * An ensemble of EnhancedLinearPerceptron classifiers
 * @author EKillick
 */
public class LinearPerceptronEnsemble extends AbstractClassifier {
    private int ensembleSize;
    private EnhancedLinearPerceptron[] arrayEnsemble;
    private double attributesPercent;
    private double instancesPercent;
    
    /**
     * Default constructor for a LinearPerceptronEnsemble
     */
    public LinearPerceptronEnsemble(){
        ensembleSize = 50;
        arrayEnsemble = new EnhancedLinearPerceptron[ensembleSize];
        attributesPercent = 50;
        instancesPercent = 50;
    }
    
    /**
     * Builds a LinearPerceptronEnsemble
     * @param i
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances i) throws Exception {
        int instAttributes = (i.numAttributes() - 1);
        int instanceCount = i.numInstances();
        int numberOfAttributes = (int) Math.round((attributesPercent/100) * instAttributes);
        int numberOfInstances = (int) Math.round((instancesPercent/100) * instanceCount);
        Random rand = new Random();
        System.out.println(attributesPercent/100);
        System.out.println(numberOfAttributes);
        
        for (int c = 0; c < ensembleSize; c++){
            // reset copy of instances
            Instances instancesCopy = new Instances(i);
            Collections.shuffle(instancesCopy);
            
            //Create subset of the instance
            Instances subset = new Instances(instancesCopy, 0, numberOfInstances);
            
            Set deletedAttributes = new HashSet();
            
            for (int count = 0; count < numberOfAttributes; count++){
                int attributeToDel;
                do{
                    attributeToDel = rand.nextInt(instAttributes + 1);
                }
                while(deletedAttributes.contains(attributeToDel));
                deletedAttributes.add(attributeToDel);
                System.out.println(deletedAttributes);

            }
            
            for (int att = 0; att < numberOfAttributes; att++){
//                System.out.println(rand.nextInt(instAttributes));
            }
//            arrayEnsemble[c] = new EnhancedLinearPerceptron(trimInstances);
            
        }
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
}
