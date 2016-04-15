/**
 * Copyright (c) 2014 ALFA Group
 * 
 * Licensed under the MIT License.
 * 
 * See the "LICENSE" file for a copy of the license.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.  
 *
 * @author Ignacio Arnaldo
 * @Author Derek Nheiley
 */

package main;

import evofmj.algorithm.RegressionEFM;
import evofmj.evaluation.DataSizeRetreiver;
import evofmj.test.TestRegressionEFM;

import java.io.File;
import java.io.IOException;


/**
 * wrapper class to manage the train and test functionality of the EFM method
 */
public class RegEFMMenuManager {
    
    /**
     * print error message and call printUsageAndExit()
     */
	private static void printUsageAndExitWithError(String errorMsg) {
		System.err.println(errorMsg);
		printUsageAndExit();
	}
    /**
     * print usage of the EFM method and exit with -1
     */
    public static void printUsageAndExit(){
        System.err.println();
        System.err.println("USAGE:");
        System.err.println();
        System.err.println("TRAIN:");
        System.err.println("java -jar efm.jar -train path_to_data -minutes min");
        System.err.println();
        System.err.println("TEST:");
        System.err.println("java -jar efm.jar -test path_to_test_data path_to_model");
        System.err.println();
        
        System.exit(-1);
    }
    
    /**
     * parse arguments to train a EFM model
     * @param args
     */
    public static void parseRegEFMTrain(String args[]) {
        String csvTrainDataPath = null;
        double timeoutMinutes = 0;
        
        if (args.length==4) {
            csvTrainDataPath = args[1];
            if (args[2].equals("-minutes")) {
            	try {
            		timeoutMinutes = Double.valueOf(args[3]);
            	} catch (NumberFormatException e) {
            		printUsageAndExitWithError("Error: must specify the optimization time in minutes");
            	}
            } else {
        		printUsageAndExitWithError("Error: must specify the optimization time in minutes");
            } 
        } else {
            printUsageAndExitWithError("Error: wrong number of arguments");
        }
        
        try { 
        	trainModel(csvTrainDataPath, timeoutMinutes);
        } catch (IOException e) {
        	printUsageAndExitWithError("Error: could not read file " + csvTrainDataPath);
        }

        System.out.println();
    }

    /**
     * parse arguments to test an EFM model
     * @param args
     * @throws IOException 
     * @throws IllegalStateException 
     */
    public static void parseRegEFMTest(String args[]) throws IllegalStateException, IOException {
        String csvTestDataPath = "";
        String modelPath = "";
        
        if (args.length==3){
        	csvTestDataPath = args[1];
            modelPath = args[2];
            if(!(new File(modelPath).isFile())){
            	System.out.println();
            	printUsageAndExitWithError("Error: model not found");
            }
        } else {
            printUsageAndExitWithError("Error: wrong number of arguments");
        }
        
        testModel(csvTestDataPath, modelPath);
        
        System.out.println();
    }

    /**
     * parse arguments of a call to EFM.jar from the command line
     * @param args
     * @throws IOException 
     * @throws IllegalStateException 
     */
    public static void main(String args[]) throws IllegalStateException, IOException {
        if (args.length == 0) {
            printUsageAndExitWithError("Error: too few arguments");
        } else {
            switch (args[0]) {
                case "-train":
                    parseRegEFMTrain(args);
                    break;
                case "-test":
                    parseRegEFMTest(args);
                    break;
                default:
                	printUsageAndExitWithError("Error: unknown argument");
            }
        }
    }
    
    /**
     * Run RegressionEFM to train a model on the provided training dataset
     * @param csvTrainDataPath
     * @param timeoutMinutes
     * @throws IOException
     */
	public static void trainModel(String csvTrainDataPath, double timeoutMinutes) throws IOException, IllegalStateException {
		RegressionEFM rEFM = new RegressionEFM(csvTrainDataPath);
		
		rEFM.numberOfLambdas = 20;
		rEFM.fitnessBias = RegressionEFM.R2;
		rEFM.modelSelectionBias = RegressionEFM.R2;
		rEFM.isNominalClasses = true;
		
		if (timeoutMinutes > 0) { 
			//lasso more than original features
			int numOriginalFeatures = DataSizeRetreiver.num_terminals(csvTrainDataPath);
		    rEFM.numArchiveFeatures = 3 * numOriginalFeatures;
		    rEFM.numNewFeatures = numOriginalFeatures;
		    rEFM.maxFinalFeatures = numOriginalFeatures + rEFM.numArchiveFeatures;
		}
		
		System.out.println("TRAIN MODEL:");
		rEFM.init();
		rEFM.runEFM(timeoutMinutes*60);
	}
	
	/**
	 * Run RegressionEFM to test the provided model against the test dataset
	 * @param csvTestDataPath
	 * @param modelPath
	 * @throws IOException
	 */
	public static void testModel(String csvTestDataPath, String modelPath) throws IOException, IllegalStateException {
		System.out.println("TESTING MODEL:");
		TestRegressionEFM test_rEFM = new TestRegressionEFM(csvTestDataPath, modelPath);
		
		test_rEFM.isNomicalClasses = true;
		test_rEFM.evalModel();
	}

}
