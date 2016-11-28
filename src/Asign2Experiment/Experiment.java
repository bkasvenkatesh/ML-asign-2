package Asign2Experiment;
/**
 *
 * @author Venkatesh Boddapati
 */
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import weka.classifiers.trees.J48;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Experiment {

   
    public static void main(String[] args) throws FileNotFoundException, IOException, Exception {
         BufferedReader reader = new BufferedReader(new FileReader("C:\\Users\\VenkuGamer\\Documents\\NetBeansProjects\\Assignment2\\src\\Asign2Experiment\\spambase.arff"));
 Instances data = new Instances(reader);
 reader.close();
 
 data.setClassIndex(data.numAttributes() - 1);
 Random random=new Random(1);
 data.randomize(random);        //shuffle data
 data.stratify(10);             // stratifies data which we use in stratified 10-fold cross validation
 
 
 // Arrays to store the metrics obtained on 10 folds using 3 algorithms
 double accuracy[][]= new double[12][3];
 double fmeasure[][]= new double[12][3];
 double time[][]= new double[12][3];
 double accsum=0;
 double fmsum=0;
 double tisum=0;
 
 long start;
 long end;
 // J48 crossvalidation
 for(int i=0; i<10; i++)
 {
     Instances trainData= data.trainCV(10,i,random);
     J48 tree= new J48();
     start=System.currentTimeMillis();
     tree.buildClassifier(trainData);
     end= System.currentTimeMillis();
     Instances testData= data.testCV(10, i);
     Evaluation eval= new Evaluation(testData);
     eval.evaluateModel(tree, testData);
     accuracy[i][0]=(eval.numTruePositives(1)+eval.numTrueNegatives(1))/eval.numInstances();
     fmeasure[i][0]=eval.fMeasure(1);
     time[i][0]=end-start;
     accsum+=accuracy[i][0];
     fmsum+=fmeasure[i][0];
     tisum+=time[i][0];
 }
 accuracy[10][0]=accsum/10;
 fmeasure[10][0]=fmsum/10;
 time[10][0]=tisum/10.0;
  
 System.out.println("J48 takes (s):"+(tisum/1000.0));
 
 
 //DecisionTable crossvalidation
 
 accsum=0; fmsum=0; tisum=0;
 for(int i=0; i<10; i++)
 {
     Instances trainData= data.trainCV(10,i,random);
     DecisionTable table= new DecisionTable();
     start=System.currentTimeMillis();
     table.buildClassifier(trainData);
     end= System.currentTimeMillis();
     Instances testData= data.testCV(10, i);
     Evaluation eval= new Evaluation(testData);
     eval.evaluateModel(table, testData);
     accuracy[i][1]=(eval.numTruePositives(1)+eval.numTrueNegatives(1))/eval.numInstances();
     fmeasure[i][1]=eval.fMeasure(1);
     time[i][1]=end-start;
     accsum+=accuracy[i][1];
     fmsum+=fmeasure[i][1];
     tisum+=time[i][1];
 }
 accuracy[10][1]=accsum/10;
 fmeasure[10][1]=fmsum/10;
 time[10][1]=tisum/10.0;
 
 System.out.println("DTable takes (s):"+(tisum/1000.0));
 
 //SimpleLogistic 
 
 accsum=0; fmsum=0; tisum=0;
 for(int i=0; i<10; i++)
 {
     Instances trainData= data.trainCV(10,i,random);
     SimpleLogistic sl= new SimpleLogistic();
     start=System.currentTimeMillis();
     sl.buildClassifier(trainData);
     end= System.currentTimeMillis();
     Instances testData= data.testCV(10, i);
     Evaluation eval= new Evaluation(testData);
     eval.evaluateModel(sl, testData);
     accuracy[i][2]=(eval.numTruePositives(1)+eval.numTrueNegatives(1))/eval.numInstances();
     fmeasure[i][2]=eval.fMeasure(1);
     time[i][2]=end-start;
     accsum+=accuracy[i][2];
     fmsum+=fmeasure[i][2];
     tisum+=time[i][2];
 }
 accuracy[10][2]=accsum/10;
 fmeasure[10][2]=fmsum/10;
 time[10][2]=tisum/10.0;
 
 System.out.println("SL takes (s):"+(tisum/1000.0));
 
 //calculating standard deviation
 double acdev[][]=new double[11][3]; acdev[10][0]=0; acdev[10][1]=0; acdev[10][2]=0;
 double fmdev[][]=new double[11][3]; fmdev[10][0]=0; fmdev[10][1]=0; fmdev[10][2]=0;
 double tidev[][]=new double[11][3]; tidev[10][0]=0; tidev[10][1]=0; tidev[10][2]=0;
 
 for(int i=0; i<10; i++)
     for(int j=0; j<3; j++)
     {
         acdev[i][j]=accuracy[i][j]-accuracy[10][j];
         acdev[i][j]*=acdev[i][j];
         acdev[10][j]+=acdev[i][j];
         fmdev[i][j]=fmeasure[i][j]-fmeasure[10][j];
         fmdev[i][j]*=fmdev[i][j];
         fmdev[10][j]+=fmdev[i][j];
         tidev[i][j]=time[i][j]-time[10][j];
         tidev[i][j]*=tidev[i][j];
         tidev[10][j]+=tidev[i][j];
     }
 
     for(int j=0; j<3; j++)
     {
         accuracy[11][j]=Math.sqrt((acdev[10][j]/10));
         fmeasure[11][j]=Math.sqrt((fmdev[10][j]/10));
         time[11][j]=Math.sqrt((tidev[10][j]/10));
     }
     
     
     
//     System.out.println("\n\n\n---------Training time table---------");
//     print12_4(time);
//     System.out.println("\n\n\n---------Accuracy table---------");
//     print12_4(accuracy);
//     System.out.println("\n\n\n---------F-Measure table---------");
//     print12_4(fmeasure);
 System.out.println("\n\n\n-----Friedman(Training time)-----");
 print12_8(time, -1);
 System.out.println("\n\n\n-----Friedman(Accuracy)-----");
 print12_8(accuracy, 1);
 System.out.println("\n\n\n-----Friedman(F-Measure)-----");
 print12_8(fmeasure, 1);
}
    
    public static void print12_4(double a[][])
    {
        System.out.println("Fold   J48 tree    DecisionTable   SimpleLogistic");
 for(int i=0; i<10; i++){
     System.out.printf("%2d      ", (i+1));
     for(int j=0; j<3; j++)
     {    
         System.out.printf("%2.4f        ", a[i][j]);
     }
     System.out.println();
    }
 System.out.printf("Avg     %2.4f        %2.4f        %2.4f\n",a[10][0],a[10][1],a[10][2]);
 System.out.printf("StdDev  %2.4f        %2.4f        %2.4f\n",a[11][0],a[11][1],a[11][2]);
    }
    
    public static void print12_8(double a[][], double x)
    {
        double rank[][][]= new double[11][3][2];
        rank[10][0][1]=0; rank[10][1][1]=0; rank[10][2][1]=0;
        double Rbar=2.0;
        
        //Determine the ranks 
        for(int i=0; i<10; i++)
        {
            for(int j=0; j<3; j++)
            {
                rank[i][j][0]=x*a[i][j];     
            }
            if( rank[i][0][0] > rank[i][1][0] ){
                if( rank[i][0][0] > rank[i][2][0] ){
                    rank[i][0][1] = 1;
                    if( rank[i][1][0] > rank[i][2][0] ){
                        rank[i][1][1] = 2;
                        rank[i][2][1] = 3;
                    }else{
                        rank[i][2][1] = 2;
                        rank[i][1][1] = 3;
                    }
                }else{
                    rank[i][0][1] = 2;
                    rank[i][2][1] = 1;
                    rank[i][1][1] = 3;
                }
            }else{
                if( rank[i][1][0] > rank[i][2][0] ){
                    rank[i][1][1] = 1;
                    if( rank[i][0][0] > rank[i][2][0] ){
                        rank[i][0][1] = 2;
                        rank[i][2][1] = 3;
                    }else{
                        rank[i][2][1] = 2;
                        rank[i][0][1] = 3;
                    }
                }else{
                    rank[i][1][1] = 2;
                    rank[i][2][1] = 1;
                    rank[i][0][1] = 3;
                }
            }
            rank[10][0][1]+=rank[i][0][1];
            rank[10][1][1]+=rank[i][1][1];
            rank[10][2][1]+=rank[i][2][1];
        }
        
        //Calculate Rj and step2
        double step2=0;
        for(int j=0; j<3; j++)
        {
            rank[10][j][1]/=10.0;
            step2+=((rank[10][j][1]-Rbar)*(rank[10][j][1]-Rbar));
        }
        step2*=10;
        
        //Calculate step3
        double step3=0;
        for(int i=0; i<10; i++)
            for(int j=0; j<3; j++)
                step3+=Math.pow((rank[i][j][1]-Rbar), 2);
        step3/=20;
        
        double friedman=step2/step3;
        
        System.out.println("Fold   J48 tree    DecisionTable   SimpleLogistic");
        for(int i=0; i<10; i++){
     System.out.printf("%2d    ", (i+1));
     for(int j=0; j<3; j++)
     {    
         System.out.printf("%2.4f (%1.0f)    ", x*rank[i][j][0], rank[i][j][1]);
     }
     System.out.println();
    }
 System.out.printf("Avg      %2.2f          %2.2f          %2.2f\n",rank[10][0][1],rank[10][1][1],rank[10][2][1]);
 System.out.println("Friedman statistic = "+friedman);
        
    }
}

