## Multi-task Learning for Multi-modal Emotion Recognition and Sentiment Analysis
Code for the paper [Multi-task Learning for Multi-modal Emotion Recognition and Sentiment Analysis](https://www.aclweb.org/anthology/N19-1034/) (NAACL 2019)

For the evaluation of our proposed multi-task CIM framerwork, we use benchmark multi-modal dataset i.e, MOSEI which has both sentiment and emotion classes.

### Dataset

* You can download datasets from [here](https://drive.google.com/open?id=1kq4_WqW0tDzBLu01yZbvdCpQ0iPBJWyQ).

* Download the dataset from given link and set the path in the code accordingly make two folders (i) results and (ii) weights.

-------------------------------------------------------
### For MOSEI Dataset:
For trimodal-->>  python trimodal_multitask.py  

-------------------------------------------------------

### Emotion Results Extractor

Follow these steps to extract the threshold based results for emotion:

* Open the text file i.e., multiTask_emotion_results_extractor.txt
* Copy and paste on the terminal

#### Example: for trimodal
##### For preference F1 score:

If the result file name is trimodal_emo.txt then run the following command 

* cat trimodal_emo.txt |  grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5' >> Emotion-Multi-task.txt

So based on threshold, desired output will be stored in Emotion-Multi-task.txt (preference is F1-score)

##### For preference W-Acc:

If the result file name is trimodal_emo.txt then run the following command 

* cat trimodal_emo.txt |  grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'6' >> Emotion-Multi-task.txt

So based on threshold, desired output will be stored in Emotion-Multi-task.txt (preference is W-Acc)

-------------------------------------------------------

### --versions--

python: 2.7

keras: 2.2.2

tensorflow: 1.9.0
