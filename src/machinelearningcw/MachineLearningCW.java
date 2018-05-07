package machinelearningcw;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.VotedPerceptron;
import java.util.Random;
import weka.core.Instance;

/**
 *
 * @author EKillick
 */
public class MachineLearningCW {
    private static int refactorCount = 10;
    private static double[][] confusionMatrix;
    
    public static void main(String[] args) {
        ArrayList<Classifier> classifierArray = new ArrayList<>();
        String[] labels = new String[7];
        
        EnhancedLinearPerceptron offlineLinearPerceptron = new EnhancedLinearPerceptron(false);
        classifierArray.add(offlineLinearPerceptron);
        labels[0] = "Offline Linear Perceptron";

        EnhancedLinearPerceptron onlineLinearPerceptron = new EnhancedLinearPerceptron(true);
        classifierArray.add(onlineLinearPerceptron);
        labels[1] = "Online Linear Perceptron";

        EnhancedLinearPerceptron offlineLinearStd = new EnhancedLinearPerceptron();
        offlineLinearStd.setStandardisation(false);
        classifierArray.add(offlineLinearStd);
        labels[2] = "Offline Linear Perceptron (std)";

        EnhancedLinearPerceptron onlineLinearStd = new EnhancedLinearPerceptron();
        classifierArray.add(onlineLinearStd);
        labels[3] = "Online Linear Perceptron (std)";

        EnhancedLinearPerceptron modelSelection = new EnhancedLinearPerceptron(true, true);
        classifierArray.add(modelSelection);
        labels[4] = "Linear Perceptron w/ Model Selection";

        MultilayerPerceptron wekaMultilayer = new MultilayerPerceptron();
        classifierArray.add(wekaMultilayer);
        labels[5] = "Weka Multilayer Perceptron";

        VotedPerceptron wekaVoted = new VotedPerceptron();
        classifierArray.add(wekaVoted);
        labels[6] = "Weka Voted Perceptron";
        
        File dir = new File("datasets");
        String[] fileNames = dir.list();        
        for (String f : fileNames) {
            System.out.println(f);
//            String dataset = "datasets/musk-1.arff";
    //        String dataset = "test_data.arff";
            Instances loadedData = loadData("datasets/" + f);
            loadedData.setClassIndex(loadedData.numAttributes() - 1);
            
    //        Instances dataArray[] = refactor(loadedData);
    //        Instances trainData = dataArray[0];
    //        Instances testData = dataArray[1];


            double prediction;
            double actual;

            for (int i = 0; i < classifierArray.size(); i++) {
                //reset for each classifier
                confusionMatrix = new double[2][2];
                for (int r = 0; r < refactorCount; r++){
                    Instances dataArray[] = refactor(loadedData);
                    Instances trainData = dataArray[0];
                    Instances testData = dataArray[1];
                    try{
                        classifierArray.get(i).buildClassifier(trainData);
                        for (Instance inst: testData){
                            prediction = classifierArray.get(i).classifyInstance(inst);
                            actual = inst.classValue();
                            confusionMatrix[(int)actual][(int)prediction]++;
                        }
                    }
                    catch(Exception error){
                        error.printStackTrace(System.err);
                    }
                }
                for(int rows = 0; rows < confusionMatrix.length; rows++){
                    for(int cols = 0; cols < confusionMatrix.length; cols++){
                        confusionMatrix[rows][cols] /= refactorCount;
                    }
                }
                System.out.println(labels[i] + " avg after " + refactorCount 
                        + " resamples: " + Arrays.deepToString(confusionMatrix));

            }
            System.out.println("==================================");
        }
    }
    
    private static Instances loadData(String location){
        Instances returnInstance = null;
        try{
            FileReader reader = new FileReader(location);
            returnInstance = new Instances(reader);
        }
        catch(IOException e){
            System.out.println("Exception caught:" +e);
        }
        return returnInstance;  
    }
    
    /**
     * A method to refactor a given data set
     * @param instances
     * @return 
     */
    private static Instances[] refactor(Instances instances){
        int numInstances = instances.numInstances();
        int numToSplit = (int)(instances.numInstances() / 2);
        
        Collections.shuffle(instances, new Random());
        
        Instances trainData = new Instances(instances, 0, numToSplit);
        Instances testData = new Instances(instances, numToSplit, numInstances-numToSplit);
        
//        System.out.println(trainData.size());
//        System.out.println(testData.size());

        Instances[] testTrain = new Instances[2];
        testTrain[0] = trainData;
        testTrain[1] = testData;
        return testTrain;
    }
        
}
