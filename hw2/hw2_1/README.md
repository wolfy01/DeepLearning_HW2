**Contents of the repositry:**

The main code is in the "model_seq2seq.py" file. 

The bash script "hw2_seq2seq.sh" is to be used for testing and generating the results text file i.e. "testset_output.txt".

The already trained model is in the "TrainedModels" directory.

The bleu_eval.py file is used for finding the accuracy of the model and calculating the bleu score.

Video feature folders "feat" for both training and testing have not been uploaded to the respective testing_data and training_data folders as per the instructions. 
They need to be added manually to carry out both training and testing.


**Training:**
For training, set Train = True in the main_execution() function in the model_seq2seq.py file. 
Models in this version is executed for 20 epochs


**Testing:**

For testing, the bash script "hw2_seq2seq.sh" has to be run. 
However, before that make sure that, Train = False is set in the main_execution() function in the model_seq2seq.py file.
The "hw2_seq2seq.sh" script is run with "testing_data" and testset_output.txt.

The trained model "model_batchsize_16_hiddensize_256_DP_0.3_worddim_2048.h5" has a bleu score of 0.6664588094412136. 
This information is also recorded in the final_result.csv file during the testing process.
