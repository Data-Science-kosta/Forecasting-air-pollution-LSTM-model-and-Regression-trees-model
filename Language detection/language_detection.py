import os
from collections import Counter
import re

if __name__=="__main__":
    print("\nWhich example you want to test (0,1,2,3 or 4)?")
    example=input()
    current_directory = os.getcwd()
    corpus_dir = os.path.join(current_directory,"publicDataSet","public","set",example,"corpus")
    sequences_dir = os.path.join(current_directory,"publicDataSet","public","set",example,"sequences.txt")
    data={}
    corpus_bigrams={}
    corpus_occurances={}
    counter=Counter() #used to count bigram occurances
    i=0 #used to index a language in languages
    print("\n--------EXTRACTING AND COUNTING BIGRAMS--------\n")
    for root, subfolders, files in os.walk(corpus_dir):
        if root==corpus_dir: 
            languages=subfolders
        if os.path.basename(root) in languages: 
            counter.clear() #clear the counter if you start processing a new language
            i+=1 
        for file in files: # processing txt files for the current language
            f=open(os.path.join(root,file),'r',encoding='utf-8')
            tekst=f.read()
            tekst=tekst.lower()
            pattern=re.compile(r'(?=(..))',flags=re.IGNORECASE) #define regex pattern
            bigrams = re.findall(pattern,tekst) #find all bigrams with given pattern
            counter.update(bigrams) #update the bigram count, with the count from the current txt file
            f.close()
        if not subfolders: 
            data[languages[i-1]]=list(counter.items())
            corpus_bigrams[languages[i-1]]=list(counter.keys())
            corpus_occurances[languages[i-1]]=list(counter.values())
    # sorting the data
    languages.sort()
    sliced_data={}
    for lg in languages:
        data[lg].sort(key=lambda tup: tup[0])
        data[lg].sort(key=lambda tup: tup[1], reverse=True)
        sliced_data[lg]=data[lg][0:5]
        # printing the language,bigram and count 
        for tuples in sliced_data[lg]:                                                                                                                                                                                
            print(lg+','+tuples[0]+f',{tuples[1]}')
    f.close()
    print("\n--------Bigrams successfully extracted--------\n")
    
    #SECOND PART, Calculating the probabilities for given sequences
    print("\n--------CALCULATING PROBABILITIES--------\n")
    f2=open(sequences_dir,'r',encoding='utf-8')
    sequences=f2.read().splitlines()
    #evaluating P(Li|text)=P(text|Li)/sum_for_every_i(P(text|Li))
    #this formula is correct only if apriori probabilities P(Li) are all the same (for every i)
    P_text_c_Li={}
    data2={}
    counter.clear()
    for lg in languages:
        P_text_c_Li[lg]=[]
        all_occur=sum(o for b,o in data[lg]) # sum of all occurances for given language
        for lines in sequences:
            lines.strip()
            lines=lines.lower()
            pattern2=re.compile(r"(?=(..))",flags=re.IGNORECASE) # define pattern for bigrams
            bigrams2 = re.findall(pattern2,lines)  # find all bigrams for given pattern
            counter.update(bigrams2)
            data2[lg]=list(counter.items()) 
            probability=1
            for (bigram,occur) in data2[lg]:
                if bigram in corpus_bigrams[lg]: # if bigram exist in the corpus
                    bigram_occur_in_corpus=corpus_occurances[lg][corpus_bigrams[lg].index(bigram)]
                    P_xy_c_Li=bigram_occur_in_corpus/all_occur # P(xy|Li)
                    probability*=P_xy_c_Li*occur
                else: 
                    probability=0 
            counter.clear() 
            P_text_c_Li[lg].append(probability) # P(text|Li)   
    for i in range(0,len(sequences)):
        print(f"Probabilities for the {i+1}. sequence:")
        for j in range(0,len(languages)):
            print(languages[j]+',',end='')
            total_probability=sum(x[i] for x in P_text_c_Li.values())# sum_for_every_i(P(text|Li))
            if P_text_c_Li[languages[j]][i]==0: 
                print(f'{0}')
            else:
                print(f'{P_text_c_Li[languages[j]][i]/total_probability}')




