package evofmj.evaluation;

/**
 * A condensed version of the weka Evaluation class used for its confusion
 * matrix code.
 * @source weka.classifiers.Evaluation $Revision: 1.53.2.6 $
 * @author derek nheiley
 */
public class ConfusionMatrix {
	
	  /** The number of classes. */
	  protected int m_NumClasses;
	
	  /** Array for storing the confusion matrix. */
	  protected double [][] m_ConfusionMatrix;
	  
	  /** The names of the classes. */
	  protected String [] m_ClassNames;
	  
	  public ConfusionMatrix(int numClasses, String [] classNames) {
		  m_NumClasses = numClasses;
		  m_ClassNames = classNames;
		  m_ConfusionMatrix = new double [m_NumClasses][m_NumClasses];
	  }

	public void set(int target, int prediction) {
		m_ConfusionMatrix[target-1][prediction-1] += 1;
	}
	
	public void printOverallAccuracy() {
		double correct = 0, total = 0;
		for (int i=0; i<m_NumClasses; i++) {
			for (int j = 0; j < m_NumClasses; j++) {
				if (j == i) {
					correct += m_ConfusionMatrix[i][j];
				}
				total += m_ConfusionMatrix[i][j];
			}
		}
		
		double totalCorrect = 0.0;
		if (total != 0) {
			totalCorrect = correct/total;
			totalCorrect *= 100.0;
		}
		
		System.out.printf("Correctly Classified Instances   %d \t %.4f %%\n", (int)correct, totalCorrect);
		System.out.printf("Incorrectly Classified Instances %d \t %.4f %%\n", (int)(total-correct), 100 - totalCorrect);
	}

	public void printAccuracyByClass() {
	    System.out.println("\n=== Detailed Accuracy By Class ===\n");
	    
	    System.out.println("TP Rate  FP Rate  Precision  Recall  Class");
	    for (int i=0; i<m_NumClasses; i++) {
	    	double truePositiveRate = getTruePositiveRate(i);
			System.out.printf("%.3f    %.3f    %.3f      %.3f   %s\n",
	    			truePositiveRate, getFalsePositiveRate(i),
	    			getPrecision(i), truePositiveRate, m_ClassNames[i]);
	    }
	}
	
  /**
   * Calculate the true positive rate with respect to a particular class. 
   * This is defined as<p/>
   * <pre>
   * correctly classified positives
   * ------------------------------
   *       total positives
   * </pre>
   *
   * @param classIndex the index of the class to consider as "positive"
   * @return the true positive rate
   */
  public double getTruePositiveRate(int classIndex) {

    double correct = 0, total = 0;
    for (int j = 0; j < m_NumClasses; j++) {
      if (j == classIndex) {
    	  correct += m_ConfusionMatrix[classIndex][j];
      }
      total += m_ConfusionMatrix[classIndex][j];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }
  
  /**
   * Calculate the false positive rate with respect to a particular class. 
   * This is defined as<p/>
   * <pre>
   * incorrectly classified negatives
   * --------------------------------
   *        total negatives
   * </pre>
   *
   * @param classIndex the index of the class to consider as "positive"
   * @return the false positive rate
   */
  public double getFalsePositiveRate(int classIndex) {

    double incorrect = 0, total = 0;
    for (int i = 0; i < m_NumClasses; i++) {
    	if (i != classIndex) {
    		for (int j = 0; j < m_NumClasses; j++) {
    			if (j == classIndex) {
    				incorrect += m_ConfusionMatrix[i][j];
    			}
    		  total += m_ConfusionMatrix[i][j];
    		}
    	}
    }
    if (total == 0) {
    	return 0;
    }
    return incorrect / total;
  }
  
  /**
   * Calculate the precision with respect to a particular class. 
   * This is defined as<p/>
   * <pre>
   * correctly classified positives
   * ------------------------------
   *  total predicted as positive
   * </pre>
   *
   * @param classIndex the index of the class to consider as "positive"
   * @return the precision
   */
  public double getPrecision(int classIndex) {

    double correct = 0, total = 0;
    for (int i = 0; i < m_NumClasses; i++) {
      if (i == classIndex) {
    	  correct += m_ConfusionMatrix[i][classIndex];
      }
      total += m_ConfusionMatrix[i][classIndex];
    }
    if (total == 0) {
      return 0;
    }
    return correct / total;
  }
  
  /**
   * Outputs the performance statistics as a classification confusion
   * matrix. For each class value, shows the distribution of 
   * predicted class values.
   *
   * @param title the title for the confusion matrix
   * @return the confusion matrix as a String
   * @throws Exception if the class is numeric
   */
  public void print(String title) {

    StringBuffer text = new StringBuffer();
    char [] IDChars = {'a','b','c','d','e','f','g','h','i','j',
		       'k','l','m','n','o','p','q','r','s','t',
		       'u','v','w','x','y','z'};
    int IDWidth;
    boolean fractional = false;

    // Find the maximum value in the matrix
    // and check for fractional display requirement 
    double maxval = 0;
    for(int i = 0; i < m_NumClasses; i++) {
      for(int j = 0; j < m_NumClasses; j++) {
	double current = m_ConfusionMatrix[i][j];
        if (current < 0) {
          current *= -10;
        }
	if (current > maxval) {
	  maxval = current;
	}
	double fract = current - Math.rint(current);
	if (!fractional
	    && ((Math.log(fract) / Math.log(10)) >= -2)) {
	  fractional = true;
	}
      }
    }

    IDWidth = 1 + Math.max((int)(Math.log(maxval) / Math.log(10) 
				 + (fractional ? 3 : 0)),
			     (int)(Math.log(m_NumClasses) / 
				   Math.log(IDChars.length)));
    text.append(title).append("\n");
    for(int i = 0; i < m_NumClasses; i++) {
      if (fractional) {
    	  text.append(" ").append(num2ShortID(i,IDChars,IDWidth - 3))
          .append("   ");
      } else {
    	  text.append(" ").append(num2ShortID(i,IDChars,IDWidth));
      }
    }
    text.append("   <-- classified as\n");
    for(int i = 0; i< m_NumClasses; i++) { 
      for(int j = 0; j < m_NumClasses; j++) {
    	  text.append(" ").append(
		    doubleToString(m_ConfusionMatrix[i][j],
					 IDWidth,
					 (fractional ? 2 : 0)));
      }
      text.append(" | ").append(num2ShortID(i,IDChars,IDWidth))
        .append(" = ").append(m_ClassNames[i]).append("\n");
    }
    System.out.println(text.toString());
  }
  
  /**
   * Method for generating indices for the confusion matrix.
   *
   * @param num 	integer to format
   * @param IDChars	the characters to use
   * @param IDWidth	the width of the entry
   * @return 		the formatted integer as a string
   */
  protected String num2ShortID(int num,char [] IDChars,int IDWidth) {
    
    char ID [] = new char [IDWidth];
    int i;
    
    for(i = IDWidth - 1; i >=0; i--) {
      ID[i] = IDChars[num % IDChars.length];
      num = num / IDChars.length - 1;
      if (num < 0) {
	break;
      }
    }
    for(i--; i >= 0; i--) {
      ID[i] = ' ';
    }

    return new String(ID);
  }
  
  /**
   * Rounds a double and converts it into String.
   *
   * @param value the double value
   * @param afterDecimalPoint the (maximum) number of digits permitted
   * after the decimal point
   * @return the double as a formatted string
   */
  public static /*@pure@*/ String doubleToString(double value, int afterDecimalPoint) {
    
    StringBuffer stringBuffer;
    double temp;
    int dotPosition;
    long precisionValue;
    
    temp = value * Math.pow(10.0, afterDecimalPoint);
    if (Math.abs(temp) < Long.MAX_VALUE) {
      precisionValue = 	(temp > 0) ? (long)(temp + 0.5) 
                                   : -(long)(Math.abs(temp) + 0.5);
      if (precisionValue == 0) {
	stringBuffer = new StringBuffer(String.valueOf(0));
      } else {
	stringBuffer = new StringBuffer(String.valueOf(precisionValue));
      }
      if (afterDecimalPoint == 0) {
	return stringBuffer.toString();
      }
      dotPosition = stringBuffer.length() - afterDecimalPoint;
      while (((precisionValue < 0) && (dotPosition < 1)) ||
	     (dotPosition < 0)) {
	if (precisionValue < 0) {
	  stringBuffer.insert(1, '0');
	} else {
	  stringBuffer.insert(0, '0');
	}
	dotPosition++;
      }
      stringBuffer.insert(dotPosition, '.');
      if ((precisionValue < 0) && (stringBuffer.charAt(1) == '.')) {
	stringBuffer.insert(1, '0');
      } else if (stringBuffer.charAt(0) == '.') {
	stringBuffer.insert(0, '0');
      }
      int currentPos = stringBuffer.length() - 1;
      while ((currentPos > dotPosition) &&
	     (stringBuffer.charAt(currentPos) == '0')) {
	stringBuffer.setCharAt(currentPos--, ' ');
      }
      if (stringBuffer.charAt(currentPos) == '.') {
	stringBuffer.setCharAt(currentPos, ' ');
      }
      
      return stringBuffer.toString().trim();
    }
    return new String("" + value);
  }
  
  /**
   * Rounds a double and converts it into a formatted decimal-justified String.
   * Trailing 0's are replaced with spaces.
   *
   * @param value the double value
   * @param width the width of the string
   * @param afterDecimalPoint the number of digits after the decimal point
   * @return the double as a formatted string
   */
  public static /*@pure@*/ String doubleToString(double value, int width,
				      int afterDecimalPoint) {
    
    String tempString = doubleToString(value, afterDecimalPoint);
    char[] result;
    int dotPosition;

    if ((afterDecimalPoint >= width) 
        || (tempString.indexOf('E') != -1)) { // Protects sci notation
      return tempString;
    }

    // Initialize result
    result = new char[width];
    for (int i = 0; i < result.length; i++) {
      result[i] = ' ';
    }

    if (afterDecimalPoint > 0) {
      // Get position of decimal point and insert decimal point
      dotPosition = tempString.indexOf('.');
      if (dotPosition == -1) {
	dotPosition = tempString.length();
      } else {
	result[width - afterDecimalPoint - 1] = '.';
      }
    } else {
      dotPosition = tempString.length();
    }
    

    int offset = width - afterDecimalPoint - dotPosition;
    if (afterDecimalPoint > 0) {
      offset--;
    }

    // Not enough room to decimal align within the supplied width
    if (offset < 0) {
      return tempString;
    }

    // Copy characters before decimal point
    for (int i = 0; i < dotPosition; i++) {
      result[offset + i] = tempString.charAt(i);
    }

    // Copy characters after decimal point
    for (int i = dotPosition + 1; i < tempString.length(); i++) {
      result[offset + i] = tempString.charAt(i);
    }

    return new String(result);
  }
	  
}
