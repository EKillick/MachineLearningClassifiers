package machinelearningcw;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;
import weka.core.DenseInstance;

/**
 * An ensemble of EnhancedLinearPerceptron classifiers
 * @author EKillick
 */
public class LinearPerceptronEnsemble extends AbstractClassifier {
    private final int ensembleSize;
    private final EnhancedLinearPerceptron[] arrayEnsemble;
    private final double attributesPercent;
    private final double instancesPercent;
    private int[][] classAttPair;
    int subsetAttributes;
    int subsetInstances;
    
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
        
        //Number of attributes and instances to keep
        subsetAttributes = (int) Math.round((attributesPercent/100) * instAttributes);
        subsetInstances = (int) Math.round((instancesPercent/100) * instanceCount);
        
        List<Integer> deletedAttributes;
        
        Random rand = new Random();
//        System.out.println(attributesPercent/100);
//        System.out.println(numberOfAttributes);

        classAttPair = new int[ensembleSize][];  
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
            classAttPair[c] = deletedAttributes.stream().mapToInt(a -> a).toArray();
            
            //Delete attributes
            for (int count = 0; count < subsetAttributes; count++){
                subset.deleteAttributeAt(classAttPair[c][count]);
            }
            
            //Store classifiers in array
            EnhancedLinearPerceptron classifier = new EnhancedLinearPerceptron(false, false);
            classifier.buildClassifier(subset);
            arrayEnsemble[c] = classifier;   
        }        
                    
        for (int inst = 0; inst < i.numInstances(); inst++){
            Instance instance = i.get(inst);
            distributionForInstance(instance);
        }
//            System.out.println(inst + " " + classifyInstance(instance)); 
    }

    /**
     * 
     * @param instnc
     * @return
     * @throws Exception 
     */
    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        double[] distForInst = distributionForInstance(instnc);
        if (distForInst[0] > distForInst[1]){
            return 0.0;
        }
        else{
            return 1.0;
        }
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        //For each classifier
        double result;
        int zeroCount = 0;
        int oneCount = 0;
        double[] groupResult = new double[2];
        for (int i = 0; i < arrayEnsemble.length; i++){
            DenseInstance instCopy = new DenseInstance(instnc); //reset instance
            //For each attribute to delete
            for (int j = 0; j < classAttPair[i].length; j++){
                instCopy.deleteAttributeAt(classAttPair[i][j]);
            }
            
            result = arrayEnsemble[i].classifyInstance(instCopy);
            
            if (result == 0.0){
                zeroCount++;
            }
            else{
                oneCount++;
            }
        }
        groupResult[0] = zeroCount;
        groupResult[1] = oneCount;
        System.out.println(groupResult[0] + " " + groupResult[1]);
        return groupResult;
    }
    
}
