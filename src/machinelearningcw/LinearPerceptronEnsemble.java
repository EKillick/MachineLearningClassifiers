package machinelearningcw;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;
import java.util.Set;

/**
 * An ensemble of EnhancedLinearPerceptron classifiers
 * @author EKillick
 */
public class LinearPerceptronEnsemble extends AbstractClassifier {
    private int ensembleSize;
    private EnhancedLinearPerceptron[] arrayEnsemble;
    private double attributesPercent;
    private double instancesPercent;
    private int[][] instAttPair;
    
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
        int subsetAttributes = (int) Math.round((attributesPercent/100) * instAttributes);
        int subsetInstances = (int) Math.round((instancesPercent/100) * instanceCount);
        List<Integer> deletedAttributes;
        Random rand = new Random();
//        System.out.println(attributesPercent/100);
//        System.out.println(numberOfAttributes);

        instAttPair = new int[ensembleSize][];  
        ensembleSize = 10;
        //For each classifier
        for (int c = 0; c < ensembleSize; c++){
            // reset copy of instances
            Instances instancesCopy = new Instances(i);
            Collections.shuffle(instancesCopy);
            
            //Create subset of the instance
            Instances subset = new Instances(instancesCopy, 0, subsetInstances);
            
            deletedAttributes = new ArrayList();
            int attributeToDel;
            
            //Select attributes to delete
            for (int count = 0; count < subsetAttributes; count++){
                do{
                    attributeToDel = rand.nextInt(instAttributes);
                }
                while(deletedAttributes.contains(attributeToDel));
                deletedAttributes.add(attributeToDel);
            }
            Collections.sort(deletedAttributes, Collections.reverseOrder());
            instAttPair[c] = deletedAttributes.stream().mapToInt(a -> a).toArray();
            
            //Delete attributes
            for (int count = 0; count < subsetAttributes; count++){
                subset.deleteAttributeAt(instAttPair[c][count]);
            }
            System.out.println(subset.toString());
            
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
