package machinelearningcw;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
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
    
    public static void main(String[] args) throws FileNotFoundException{
        ArrayList<Classifier> classifierArray = new ArrayList<>();
        String[] labels = new String[8];
        
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
        
        LinearPerceptronEnsemble linearEnsemble = new LinearPerceptronEnsemble();
        classifierArray.add(linearEnsemble);
        labels[4] = "Linear Perceptron Ensemble";

        EnhancedLinearPerceptron modelSelection = new EnhancedLinearPerceptron(true, true);
        classifierArray.add(modelSelection);
        labels[5] = "Linear Perceptron w/ Model Selection";

        MultilayerPerceptron wekaMultilayer = new MultilayerPerceptron();
        classifierArray.add(wekaMultilayer);
        labels[6] = "Weka Multilayer Perceptron";

        VotedPerceptron wekaVoted = new VotedPerceptron();
        classifierArray.add(wekaVoted);
        labels[7] = "Weka Voted Perceptron";
        
        File dir = new File("datasets");
        PrintWriter output = new PrintWriter(new File("results.csv"));
        StringBuilder sb = new StringBuilder();
        sb.append("Dataset,");
        sb.append("Classifier,");
        sb.append("TP,");
        sb.append("FP,");
        sb.append("FN,");
        sb.append("TN,");
        sb.append("\n");
        
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
                sb.append(labels[i]).append(",");
                sb.append(f).append(",");
                for(int rows = 0; rows < confusionMatrix.length; rows++){
                    for(int cols = 0; cols < confusionMatrix.length; cols++){
                        confusionMatrix[rows][cols] /= refactorCount;
                        sb.append(confusionMatrix[rows][cols]).append(",");
                    }
                }
                sb.append("\n");
                System.out.println(labels[i] + " avg after " + refactorCount 
                        + " resamples: " + Arrays.deepToString(confusionMatrix));

            }
            sb.append('\n');
            System.out.println("==================================");
        }
        output.write(sb.toString());
        output.close();

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

        Instances[] testTrain = new Instances[2];
        testTrain[0] = trainData;
        testTrain[1] = testData;
        return testTrain;
    }
        
}
