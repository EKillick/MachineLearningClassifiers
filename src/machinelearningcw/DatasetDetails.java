package machinelearningcw;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import weka.core.Instances;
import weka.core.Instance;

/**
 *
 * @author EKillick
 */
public class DatasetDetails {

    public static void main(String[] args) throws FileNotFoundException {

        File dir = new File("datasets");
        PrintWriter output = new PrintWriter(new File("datasets.csv"));
        StringBuilder sb = new StringBuilder();
        sb.append("Dataset,");
        sb.append("Attributes,");
        sb.append("Instances,");
        sb.append("Class 0 Count,");
        sb.append("Class 1 Percentage,");
        sb.append("Class 1 Count,");
        sb.append("Class 1 Percentage,");
        sb.append("\n");

        String[] fileNames = dir.list();
        for (String f : fileNames) {
            double classZero = 0;
            double classOne = 0;
            double classZeroPercent;
            Instances loadedData = loadData("datasets/" + f);
            loadedData.setClassIndex(loadedData.numAttributes() - 1);
            sb.append(f).append(",");
            sb.append(loadedData.numAttributes()-1).append(",");
            sb.append(loadedData.numInstances()).append(",");
            
            for (Instance i : loadedData){
                if(i.classValue() == 0){
                    classZero++;
                }
                else{
                    classOne++;
                }
            }
            
            classZeroPercent = (classZero / loadedData.numInstances())*100;
            
            sb.append(classZero).append(",");
            sb.append(classZeroPercent).append(",");
            sb.append(classOne).append(",");
            sb.append(100-classZeroPercent).append(",");
            sb.append("\n");
        }
        output.write(sb.toString());
        output.close();
    }

    private static Instances loadData(String location) {
        Instances returnInstance = null;
        try {
            FileReader reader = new FileReader(location);
            returnInstance = new Instances(reader);
        } catch (IOException e) {
            System.out.println("Exception caught:" + e);
        }
        return returnInstance;
    }

}
