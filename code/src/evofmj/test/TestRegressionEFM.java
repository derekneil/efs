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
package evofmj.test;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import evofmj.evaluation.ConfusionMatrix;
import evofmj.evaluation.java.EFMScaledData;
import evofmj.genotype.Tree;
import evofmj.genotype.TreeGenerator;
import evofmj.math.Function;

/**
 * Implements fitness evaluation for symbolic regression.
 */
public class TestRegressionEFM {
    
    private EFMScaledData testData;    
    private List<Tree> features; 
    private List<Double> weights;
    private double lassoIntercept = 0;
    public double minTarget, maxTarget;
    public boolean isNomicalClasses = false;
    public ConfusionMatrix matrix;
    
    /**
     * Test models obtained with the EFM method
     * Complex features are mapped to expression trees, evaluated via the 
     * standard inorder parsing used in tree-based Genetic Programming
     * @param csvTestDataPath
     * @param pathToModel
     * @throws java.io.IOException
     * @throws java.lang.ClassNotFoundException
     */
    public TestRegressionEFM( String csvTestDataPath, String pathToModel) throws IOException {
        testData = new EFMScaledData(csvTestDataPath);
        features = new ArrayList<Tree>(); 
        weights = new ArrayList<Double>();
        lassoIntercept = 0;
        readModel(pathToModel);
    }

    /**
    * auxiliary class to read a model from file
    */
    private void readModel(String pathToModel) throws IOException {
        
        Scanner sc = new Scanner(new FileReader(pathToModel));
        
        String lineMinMax = sc.nextLine();
        String[] minMax = lineMinMax.split(",");
        minTarget = Double.valueOf(minMax[0]);
        maxTarget = Double.valueOf(minMax[1]);
        
        lassoIntercept = Double.valueOf(sc.nextLine());
        
        while(sc.hasNextLine()){
            String sAux = sc.nextLine();
            sAux = sAux.trim();
            String[] tokens = sAux.split(" ");
            double wAux = Double.valueOf(tokens[1]);
            weights.add(wAux);
            
            String featureStringAux = "";
            for(int i=3;i<tokens.length;i++){
                featureStringAux += tokens[i] + " ";
            }
            featureStringAux = featureStringAux.trim();
            Tree g = TreeGenerator.generateTree(featureStringAux);
            features.add(g);
        }
        sc.close();
    }
    
   
    /**
     * @see eval an EFM model
     */
    public void evalModel() {
        double[][] inputValuesAux = testData.getInputValues();
        double[] targets = testData.getTargetValues();
        double sqDiff = 0;
        double absDiff = 0;
        
        int cases = testData.getNumberOfFitnessCases();
        int classes = 0;
        if (isNomicalClasses) {
        	classes = testData.getNumberOfDistinctTargetValues();
        	matrix = new ConfusionMatrix(classes, testData.getTargetStrings());
        }
        	
        double startTime = System.currentTimeMillis();
        
		for (int i = 0; i < cases; i++) {
        	double prediction = lassoIntercept;
            List<Double> d = new ArrayList<Double>();
            for (int j = 0; j < testData.getNumberOfOriginalFeatures(); j++) {
                d.add(j, (double)inputValuesAux[i][j]);
            }
            
            for (int j = 0; j < features.size(); j++) {
                Tree genotype = (Tree) features.get(j);
                Function func = genotype.generate();
                double funcOutput = func.eval(d);
                if(Double.isNaN(funcOutput) || Double.isInfinite(funcOutput)){
                    funcOutput=0;
                }
                if(weights.get(j)!=0){
                    prediction += weights.get(j) * funcOutput;
                }
            }
            
            //TODO isn't this kind of cheating for predicting the min and max targets?
            if (prediction<minTarget) { prediction = minTarget; }
            if (prediction>maxTarget) { prediction = maxTarget; }
            
            //roundPrediction can be used for nominal classes represented by integers
            if (isNomicalClasses) { prediction = Math.round(prediction); }
            
            d.clear();
            double diff = targets[i] - prediction;
			sqDiff += Math.pow(diff, 2);
            absDiff += Math.abs(diff);
            
            if (isNomicalClasses) {
            	matrix.set((int)targets[i], (int)prediction);
            }
        }
		
		double evaluationTime = (System.currentTimeMillis() - startTime)/1000.0;
		sqDiff = sqDiff / cases;
        absDiff= absDiff / cases;
        
        System.out.printf("Time taken to evaluate model: %.2f seconds\n", evaluationTime);
        System.out.println("\n=== Summary ===\n");
        
        if (isNomicalClasses) {
        	matrix.printOverallAccuracy();
        }
        System.out.printf("Mean squared error                       %.4f\n", sqDiff);
        System.out.printf("Root mean squared error                  %.4f\n", Math.sqrt(sqDiff));
        System.out.printf("Mean absolute error                      %.4f\n", absDiff);
        
        System.out.printf("Total Number of Instances         %d\n", cases);
        
        if (isNomicalClasses) {
        	matrix.printAccuracyByClass();
	        matrix.print("\n=== Confusion Matrix ===\n");
        }
    }

}