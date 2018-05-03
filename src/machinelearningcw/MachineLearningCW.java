package machinelearningcw;

import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;
/**
 *
 * @author EKillick
 */
public class MachineLearningCW {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String testLocation = "test_data.arff";
        Instances testInstance = loadData(testLocation);
        testInstance.setClassIndex(testInstance.numAttributes() - 1);
        EnhancedLinearPerceptron classTest = new EnhancedLinearPerceptron(true, true);
//        LinearPerceptron classTest = new LinearPerceptron();
        try{
            classTest.buildClassifier(testInstance);
        }
        catch(Exception error){
            error.printStackTrace(System.err);
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
    
}
