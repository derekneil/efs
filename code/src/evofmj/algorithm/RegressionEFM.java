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

package evofmj.algorithm;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;
import evofmj.evaluation.java.EFMScaledData;

/**
 * Main class of the Evolutionary Feature Search Method
 */
public class RegressionEFM {
    
    final boolean VERBOSE = false;
    private long startTime;
    private double timeout = 0;
    private Random rand;
    private boolean initialized = false;
    
    public int numOriginalFeatures = -1;
    public int numArchiveFeatures = 0;
    public int numNewFeatures = 0;
    public int maxFinalFeatures = numOriginalFeatures;
    public int maxFeatureSize = 5;
    
    public int numberOfLambdas = LassoFitGenerator.MIN_NUMBER_OF_LAMBDAS;
    public int TOURNAMENT_SIZE = 2;
    final double BINARY_RECOMB_RATE = 0.5;
    
    private String csvPath;
    private EFMScaledData dataMatrix;
    private double[] featureScores;
    private int[] indicesArchive;
    private List<String> weights;

    /** targets will be rounded to nearest integer when nominal is true **/
    public boolean isNominalClasses = false;
    public double minTarget, maxTarget;
    private double bestMSE = Double.MAX_VALUE;

    final String[] UNARY_OPS = {"mylog","exp","mysqrt","square","cube","cos","sin"};
    final String[] BINARY_OPS = {"*","mydiv","+","-"};

    private List<String> bestFeatures = new ArrayList<>();
    private List<Double> bestWeights = new ArrayList<>();
    private double lassoIntercept = 0;
    private double bestIntercerpt;
    
    private int stallIterations = 0;
    public int maxStallIterations = 200;
    
    public static final int VARIABLE_COUNT = 0;
    public static final int R2 = 1;
    public static final int MSE = 2;
    
    public int fitnessBias = R2;
    public int modelSelectionBias = R2;
    
    /**
     * constructor
     * @param csvPath
     * @param aNumberOfArchiveFeatures
     * @param aNumberOfNewFeatures
     * @param aMaxFeatureSize
     * @param aMaxFinalFeatures
     * @throws IOException
     */
    public RegressionEFM(String csvPath) {
    	this.csvPath = csvPath;
    }
    
    /**
     * @param seed optional parameter to control seed value for experiements
     * @throws IOException
     */
    public void init() throws IOException { init(null); }
    /**
     * Build datamatix with for original, archive, and new features, set start
     * time and initialize random generator
     * @param seed optional parameter to control seed value for experiments
     * @throws IOException
     */
    public void init(Long seed) throws IOException {
    	dataMatrix = new EFMScaledData(numArchiveFeatures, numNewFeatures, csvPath);
    	numOriginalFeatures = dataMatrix.getNumberOfOriginalFeatures();
        
        featureScores = new double[dataMatrix.getNumberOfTotalFeatures()];
        indicesArchive = new int[numArchiveFeatures];
        for (int i=0; i<numArchiveFeatures; i++) { 
        	indicesArchive[i] = numOriginalFeatures + i;
        }
        
        startTime = System.currentTimeMillis();
        System.out.println("Start Time: "+startTime);
        if (seed==null) {
        	seed = startTime;
        }
    	rand = new Random(seed);
    	System.out.println("Random Seed: "+seed);

    	initialized = true;
    }
    
    /**
     * main loop of the EFM method
     * @param timeoutInSeconds
     * @throws IllegalStateException
     */
    public void runEFM(double timeoutInSeconds) throws IOException, IllegalStateException {
    	if (!initialized) {
    		init();
    	}
    	
    	if (timeoutInSeconds > 0) {
        	timeout = startTime + (timeoutInSeconds * 1000);
        }
        
        dataMatrix.fillInitialArchiveandNewFeatures(rand);
        evalAllFeatures();
        
        int indexIteration = 0;
        do {
            generateNewFeatures();
            evalAllFeatures();
            generateModel();

            if (VERBOSE) {
                saveCurrentFeatureSet(indexIteration);
                saveCurrentModel(indexIteration);
            }
        	indexIteration++;
        } while (!stopCriteriaSatisfied());
        	
        saveBestFeatureSet(indexIteration, true);
        saveBestModel(indexIteration, true);
    }
    
    /**
     * stop criteria: convergence or timeout is reached
     * @return whether the run must be stopped
     */
    public boolean stopCriteriaSatisfied() {
        if ( timeout > 0 && System.currentTimeMillis() >= timeout) {
            System.out.println("Timout exceeded, exiting. BEST MSE IS: " + bestMSE);
            return true;
        } else if (stallIterations > maxStallIterations) {
            System.out.println("Progress Stalled, exiting. BEST MSE IS: " + bestMSE);
            return true;
        }
        return false;
    }
    
    /**
    * Estimate feature importance according to
    * the number of appearances of a feature in the regularized models obtained
    * via pathwise coordinate descent
    */
    private List<Feature> computeFeatureImportanceVariableCount(LassoFit fit) {
        int indexWeights = 0;
        double rcoeff = 0;
        for (int i=0;i<fit.lambdas.length;i++) {
            if (fit.rsquared[i]> rcoeff) {
                indexWeights = i;
                rcoeff = fit.rsquared[i];
            }
        }
        
        double[] lassoWeights = fit.getWeights(indexWeights);
        List<Feature> tempFeatureScores = new ArrayList<>();
        for (int j=0;j<lassoWeights.length;j++) {
            if (lassoWeights[j]!=0) {
                featureScores[j] = 0;
                for (int z=0;z<fit.nonZeroWeights.length;z++) {
                    double[] lassoWeightsAux = fit.getWeights(z);
                    if (lassoWeightsAux[j]!=0) featureScores[j]++;
                }
            } else{
                featureScores[j] = 0;
            }
            if (j>=numOriginalFeatures) {
                Feature fsAux = new Feature(j,featureScores[j]);
                tempFeatureScores.add(fsAux);    
            }
        }
        
        return tempFeatureScores;
    }

    /**
    * Estimate feature importance according to the coefficient of multiple 
    * correlation of the models in which the feature appears
    */
    private List<Feature> computeFeatureImportanceBiasR2(LassoFit fit) {
        int indexWeights = 0;
        //get argmax rsquared
        double rcoeff = 0;
        for (int i=0; i<fit.lambdas.length; i++) {
            if (fit.rsquared[i] > rcoeff) {
                indexWeights = i;
                rcoeff = fit.rsquared[i];
            }
        }
        
        double[] lassoWeights = fit.getWeights(indexWeights);
        List<Feature> tempFeatureScores = new ArrayList<>();
        for (int j=0; j<lassoWeights.length; j++) {
            featureScores[j] = 0;
            if (lassoWeights[j]!=0) {
                for (int z=0; z<fit.nonZeroWeights.length; z++) {
                    double[] lassoWeightsAux = fit.getWeights(z);
                    if (lassoWeightsAux[j]!=0) {
                        //featureScores[j] += fit.rsquared[z]/(dataMatrix.getFeatureSize(j)*0.5);
                        featureScores[j] += fit.rsquared[z];
                    }
                }
            }
            if (j >= numOriginalFeatures) {
                Feature fsAux = new Feature(j,featureScores[j]);
                tempFeatureScores.add(fsAux);    
            }
        }
        
        return tempFeatureScores;
    }    
    
    /**
    * Estimate feature importance according to the mean squared error of the
    * models in which the feature appears
    */
    private List<Feature> computeFeatureImportanceBiasMSE(LassoFit fit) {
        double[][] dataMatrixAux = dataMatrix.getInputValues();
        double[] targetsAux = dataMatrix.getTargetValues();
        double minSqError = Double.MAX_VALUE;
        int indexLambdaMinError = 0;
        double[] mseLambdas = new double[fit.lambdas.length];
        
        for (int l=0; l<fit.lambdas.length; l++) {    
            double interceptAux = fit.intercepts[l];
            double[] lassoWeightsAux = fit.getWeights(l);
            double sqError = 0;
            for (int i=0; i < dataMatrix.getNumberOfFitnessCases(); i++) {
                double prediction = interceptAux;
                int indexFeature=0;
                for (int j=0; j < dataMatrix.getNumberOfTotalFeatures(); j++) {
                    prediction += dataMatrixAux[i][j]*lassoWeightsAux[indexFeature];
                    indexFeature++;
                }
                sqError += Math.pow(targetsAux[i] - prediction,2);
            }
            sqError = sqError / dataMatrix.getNumberOfFitnessCases();
            mseLambdas[l] = sqError;
            if ((sqError<minSqError)) {
                minSqError = sqError;
                indexLambdaMinError = l;
            }
        }
        
        double[] lassoWeights = fit.getWeights(indexLambdaMinError);
        List<Feature> tempFeatureScores = new ArrayList<>();
        for (int j=0; j<lassoWeights.length; j++) {
            featureScores[j] = 0;
            if (lassoWeights[j]!=0) {
                for (int l=0; l<fit.numberOfLambdas; l++) {
                    double[] lassoWeightsAux = fit.getWeights(l);
                    if (lassoWeightsAux[j]!=0) {
                        featureScores[j] += 1/mseLambdas[l]; //check the bias, not sure it makes sense to add 1/mse;;;;;;
                    }
                }
            }
            if (j >= numOriginalFeatures) {
                Feature fsAux = new Feature(j, featureScores[j]);
                tempFeatureScores.add(fsAux);    
            }
        }
        
        return tempFeatureScores;
    }

    /**
    * A coordinate descent method for the Lasso is run and the resulting
    * models are mined to estimate the importance of the features.
    * 
    * @throws IllegalStateException
    */
    private void evalAllFeatures() throws IllegalStateException{
        LassoFitGenerator fitGenerator = new LassoFitGenerator();
        int numObservations = dataMatrix.getNumberOfFitnessCases();
        fitGenerator.init(dataMatrix.getNumberOfTotalFeatures(), numObservations);
        for (int i=0; i < numObservations; i++) {
            double[] row = dataMatrix.getRow(i);
            float[] rowFloats = new float[dataMatrix.getNumberOfTotalFeatures()];
            for (int j=0; j<dataMatrix.getNumberOfTotalFeatures(); j++) {
                rowFloats[j]=(float)row[j];
            }
            fitGenerator.setObservationValues(i, rowFloats);
            fitGenerator.setTarget(i, dataMatrix.getTargetValues()[i]);
        }

        LassoFit fit = fitGenerator.fit( dataMatrix.getNumberOfTotalFeatures(), 
                                         numberOfLambdas );
        
        weights = new ArrayList<String>();
        List<Feature> tempFeatureScores = null;
        
        //These are variant to estimate feature importance
        if (fitnessBias==VARIABLE_COUNT) {
        	//bias feature importance of feature with # of different Lambdas
        	tempFeatureScores =  computeFeatureImportanceVariableCount(fit);
        } else if (fitnessBias==R2) {
        	// bias feature importance of feature with r2 of different Lambdas
        	tempFeatureScores = computeFeatureImportanceBiasR2(fit);
        } else if (fitnessBias==MSE) {
        	// bias feature importance with MSE of different Lambdas
        	tempFeatureScores = computeFeatureImportanceBiasMSE(fit);
        } else {
        	throw new IllegalStateException("Unknown FITNESS_BIAS " + fitnessBias);
        }

        //populate archive with features by score
        Collections.sort(tempFeatureScores);
        for (int i=0; i<numArchiveFeatures; i++) {
            indicesArchive[i] = tempFeatureScores.get(i).getIndex();
        }
    }
    
    /**
    * Select the model that maximizes the coefficient of multiple correlation
    */
    private int getIndexLambdaModelSelectionR2(LassoFit fit) {
        int indexLambda = 0;
        double rsquared = fit.rsquared[indexLambda];
        for (int i=1; i<fit.lambdas.length; i++) {
            if (fit.rsquared[i] > rsquared) {
                indexLambda = i;
            }
        }
        return indexLambda;
    }
        
    /**
    * Select the model that minimizes the mean squared error
    * @param indexLambdaMinError
    */    
    private int getIndexLambdaModelSelectionMSE(LassoFit fit) {
        double[][] dataMatrixAux = dataMatrix.getInputValues();
        double[] targetsAux = dataMatrix.getTargetValues();
        double minSqError = Double.MAX_VALUE;
        int indexLambdaMinError = 0;
        for (int l=0; l < fit.lambdas.length; l++) {
            double interceptAux = fit.intercepts[l];
            double[] lassoWeightsAux = fit.getWeights(l);
            double sqError = 0;
            for (int i=0; i < dataMatrix.getNumberOfFitnessCases(); i++) {
                double prediction = interceptAux;
                int indexFeature =0;
                for (int j=0; j < numOriginalFeatures+numArchiveFeatures; j++) {
                    prediction += dataMatrixAux[i][j]*lassoWeightsAux[indexFeature];
                    indexFeature++;
                }
                sqError += Math.pow(targetsAux[i] - prediction,2);
            }
            sqError = sqError / dataMatrix.getNumberOfFitnessCases();
            if (sqError<minSqError) {
                minSqError = sqError;
                indexLambdaMinError = l;
            }
        }
        return indexLambdaMinError;
    }
    
    /**
    * Iniatlize LassoFitGenerator to obtain a linear model with the 
    * best/selected features and print summary statistics
    * 
    * @throws IllegalStateException
    */
    private void generateModel() throws IllegalStateException {
        LassoFitGenerator fitGenerator = new LassoFitGenerator();
        int numObservations = dataMatrix.getNumberOfFitnessCases();
        fitGenerator.init(numOriginalFeatures+numArchiveFeatures, numObservations);
        
        for (int i=0; i < numObservations; i++) {
            double[] row = dataMatrix.getRow(i);
            float[] reducedRow = new float[numOriginalFeatures+numArchiveFeatures];
            int indexAddedFeature=0;
            for (int j=0;j<dataMatrix.getNumberOfTotalFeatures();j++) {
                if (j<numOriginalFeatures || archiveContains(j) ) {
                    reducedRow[indexAddedFeature]=(float)row[j];
                    indexAddedFeature++;
                }
            }
            fitGenerator.setObservationValues(i,reducedRow);
            fitGenerator.setTarget(i, dataMatrix.getTargetValues()[i]);
        }

        LassoFit fit = fitGenerator.fit(maxFinalFeatures, numberOfLambdas);
        
        int indexLambda = 0;
        //choose a value for the regularization coefficient Lambda
        if (modelSelectionBias==R2) {
            indexLambda = getIndexLambdaModelSelectionR2(fit);
        } else if (modelSelectionBias==MSE) {
            indexLambda = getIndexLambdaModelSelectionMSE(fit);
        } else {
        	throw new IllegalStateException("Unknown MODEL_SELECTION_BIAS " + modelSelectionBias);
        }

        weights = new ArrayList<String>();
        double[] lassoWeights = fit.getWeights(indexLambda);
        
        for (int j=0; j<lassoWeights.length; j++) {
            weights.add(Double.toString(lassoWeights[j]));
        }
        lassoIntercept = fit.intercepts[indexLambda];
        
        // We compute the mean squared error of the selected model
        double[][] dataMatrixAux = dataMatrix.getInputValues();
        double[] targetsAux = dataMatrix.getTargetValues();
        double sqError = 0;
        double absError = 0;
        
        for (int i=0; i < dataMatrix.getNumberOfFitnessCases(); i++) {
            double prediction = lassoIntercept;
            int indexFeature =0;
            for (int j=0;j<dataMatrix.getNumberOfTotalFeatures();j++) {
                if (j<numOriginalFeatures || archiveContains(j) ) {
                    prediction += dataMatrixAux[i][j] * lassoWeights[indexFeature];
                    indexFeature++;
                }
            }
            
            //TODO isn't this kind of cheating for predicting the min and max targets?
//            if (prediction<minTarget) { prediction = minTarget; }
//            if (prediction>maxTarget) { prediction = maxTarget; }
            
            //roundPrediction can be used for nominal classes represented by integers
            if (isNominalClasses) { prediction = Math.round(prediction); }
            
            double diff = targetsAux[i] - prediction;
            sqError += Math.pow(diff, 2);
            absError += Math.abs(diff);
        }
        sqError  = sqError  / dataMatrix.getNumberOfFitnessCases();
        absError = absError / dataMatrix.getNumberOfFitnessCases();
        
        if (sqError < bestMSE) {
            bestMSE = sqError;
            stallIterations = 0;
            bestFeatures = new ArrayList<String>();
            bestWeights = new ArrayList<Double>();
            bestIntercerpt = lassoIntercept;
            int indexFeature = 0;
            for (int j=0; j<dataMatrix.getNumberOfTotalFeatures(); j++) {
                if (j<numOriginalFeatures || archiveContains(j) ) {
                    String featureAux = dataMatrix.getFeatureString(j);
                    double weightAux = lassoWeights[indexFeature];
                    bestFeatures.add(featureAux);
                    bestWeights.add(weightAux);
                    indexFeature++;
                }
            }
        } else{
            stallIterations++;
        }
        
        double runtime = (System.currentTimeMillis() - startTime)/1000.0;
        System.out.printf("RUNTIME: %10.2f ; MSE: %.6f ; MAE: %.6f ; BEST MSE: %.6f\n", runtime, sqError, absError, bestMSE);
    }
    
    
    /**
    * compose new features from a tournament of features ( original and/or 
    * archive) using randomly selected binary or unary operators storing
    * results in dataMatrix
    */
    private void generateNewFeatures() {    
        int indexStart = numOriginalFeatures;
        int indexEnd = dataMatrix.getNumberOfTotalFeatures();
        for (int j=indexStart; j<indexEnd; j++) {
            if (archiveContains(j)) {
            	continue;
            }
            
            int indexParent1 = tournamentSelection();
            if (rand.nextFloat()< BINARY_RECOMB_RATE) {
                int indexParent2 = tournamentSelection();
                if ((dataMatrix.getFeatureSize(indexParent1) + dataMatrix.getFeatureSize(indexParent2)) < maxFeatureSize) {
                    binaryRecombination(j,indexParent1,indexParent2);
                } else{
                    dataMatrix.setFeatureToZero(j);
                }
            } else{
                if (dataMatrix.getFeatureSize(indexParent1) < maxFeatureSize) {
                    unaryRecombination(j,indexParent1);
                } else{
                    dataMatrix.setFeatureToZero(j);
                }
            }
        }
    }
    
    /**
    * Auxiliar method to check a given feature is part of the archive already
    * @return boolean
    */
    private boolean archiveContains(int index) {
        for (int i=0;i<numArchiveFeatures;i++) {
            if (indicesArchive[i]==index) {
                return true;
            }
        }
        return false;
    }
    
    /**
    * Select feature with highest score from TOURNAMENT_SIZE of randomly 
    * selected features (original and archive of features).
    * @return indexParent
    */
    private int tournamentSelection() {
        int indexParent = rand.nextInt(numOriginalFeatures + numArchiveFeatures);
        if (indexParent>=numOriginalFeatures) {
            indexParent = indicesArchive[indexParent-numOriginalFeatures];
        }
        for (int i=0; i<TOURNAMENT_SIZE-1; i++) {
            int indexAux = rand.nextInt(numOriginalFeatures + numArchiveFeatures);
            if (indexAux>=numOriginalFeatures) {
                indexAux = indicesArchive[indexAux-numOriginalFeatures];
            }
            if (featureScores[indexAux] > featureScores[indexParent]) indexParent = indexAux;
        }
        return indexParent;
    }
    
    /**
    * composition of new features via binary functions
    */
    private void binaryRecombination(int indexNewFeature,int indexParent1,int indexParent2) {
        int indexOp  = rand.nextInt(BINARY_OPS.length);
        switch (BINARY_OPS[indexOp]) {
            case "*":
                dataMatrix.multiplication(indexNewFeature,indexParent1,indexParent2);
                break;
            case "mydiv":
                dataMatrix.division(indexNewFeature,indexParent1,indexParent2);
                break;
            case "+":
                dataMatrix.sum(indexNewFeature,indexParent1,indexParent2);
                break;
            case "-":
                dataMatrix.minus(indexNewFeature,indexParent1,indexParent2);
                break;
        }
    }

    /**
    * composition of new features via unary functions
    */
    private void unaryRecombination(int indexNewFeature,int indexParent1) {
        int indexOp  = rand.nextInt(UNARY_OPS.length);
        switch (UNARY_OPS[indexOp]) {
            case "mylog":
                dataMatrix.log(indexNewFeature,indexParent1);
                break;
            case "exp":
                dataMatrix.exp(indexNewFeature,indexParent1);
                break;
            case "mysqrt":
                dataMatrix.sqrt(indexNewFeature,indexParent1);
                break;
            case "square":
                dataMatrix.square(indexNewFeature,indexParent1);
                break;
            case "cube":
                dataMatrix.cube(indexNewFeature,indexParent1);
                break;
            case "cos":
                dataMatrix.cos(indexNewFeature,indexParent1);
                break;
            case "sin":
                dataMatrix.sin(indexNewFeature,indexParent1);
                break;
        }
    }
    
    /**
    * worker method to save a string in a file
    */
    private static void saveText(String filepath, String text, Boolean append) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(filepath,append));
        PrintWriter printWriter = new PrintWriter(bw);
        printWriter.write(text);
        printWriter.flush();
        printWriter.close();
    }
        
    /**
    * Logging method for research purposes: 
    * save the population of features during the run
    */
    private void saveCurrentFeatureSet(int indexIteration) throws IOException{
        String featuresPath = "features_" + indexIteration + ".txt";
        saveText(featuresPath,"", false);
        for (int j=0; j<dataMatrix.getNumberOfTotalFeatures(); j++) {
            if (j<numOriginalFeatures || archiveContains(j) ) {
                saveText(featuresPath, dataMatrix.getFeatureString(j) + ",", true);
            }
        }
    }
    
    /**
    * Logging method for research purposes: 
    * save the model at the end of the iteration/generation
    */
    private void saveCurrentModel(int indexIteration) throws IOException{
        System.out.println(indexIteration);
        String modelPath = "model_" + indexIteration + ".txt";
        saveText(modelPath, lassoIntercept + "\n", false);
        int indexFeature = 0;
        for (int j=0; j<dataMatrix.getNumberOfTotalFeatures(); j++) {
            if (j<numOriginalFeatures || archiveContains(j) ) {
                saveText(modelPath, " + " + weights.get(indexFeature) + " * " + dataMatrix.getFeatureString(j) + "\n", true);
                indexFeature++;
            }
        }
    }
    
    /**
    * save the population of features at the end of the run
    */
    private void saveBestFeatureSet(int indexIteration, boolean finished) throws IOException{
        String featuresPath = "features_" + indexIteration + ".txt";
        if (finished) featuresPath = "features.txt";
        saveText(featuresPath,"", false);
        for (int j=0; j<bestFeatures.size(); j++) {
            saveText(featuresPath, bestFeatures.get(j) + ",", true);
        }
    }
  
    /**
    * save the model at the end of the run
    */
    private void saveBestModel(int indexIteration, boolean finished) throws IOException{
        String modelPath = "model_" + indexIteration + ".txt";
        if (finished) modelPath = "model.txt";
        String targetRange = dataMatrix.getTargetMin() + "," + dataMatrix.getTargetMax() + "\n";
		saveText(modelPath, targetRange, false);
        String intercept = bestIntercerpt + "\n";
		saveText(modelPath, intercept, true);
        
        List<Feature> features = new ArrayList<>();
        for (int j=0; j<bestFeatures.size(); j++) {
            Double score = bestWeights.get(j);
			if (score != 0) {
				saveText(modelPath, " + " + score + " * " + bestFeatures.get(j) + "\n", true);

				String feature = " +" + score;
				if (score < 0) { 
					feature = " " + score;
					score = -score;
				}
                features.add(new Feature(score, feature + " * " + bestFeatures.get(j) + "\n"));
            }
        }
        System.out.println("\n=== Best Model ===\n");
        System.out.print(targetRange);
        System.out.print(intercept);

        Collections.sort(features);
        for (Feature feature : features) {
        	System.out.print(feature.getFeature());
        }
    }
    
    /**
    * worker class to sort features according to their estimated score/importance
    */
    private class Feature implements Comparable<Feature> {
        private int index;
        private double score;
        private String feature;
        
        public Feature(int anIndex, double aScore) {
            index = anIndex;
            score = aScore;
        }
        
        public Feature(double aScore, String aFeature) {
        	score = aScore;
        	feature = aFeature;
        }
        
        public int getIndex() { return index; }
        public double getScore() { return score; }
        public String getFeature() { return feature; }
        
        @Override
        public int compareTo(Feature other) {
            int comp = 0;
            if (this.score>other.getScore()) {
                comp=-1;
            } else if (this.score<other.getScore()) {
                comp = 1;
            }
            return comp;
        }
    }
}

