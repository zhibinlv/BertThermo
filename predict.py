import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from biotransformers import BioTransformers
import torch
import Bio
import joblib
from sklearn.linear_model import LogisticRegression
import sklearn as sk
import argparse
from biotransformers import BioTransformers
print(sk.__version__)
print(torch.__version__)
print(Bio.__version__)

from Bio import SeqIO

def read_fasta(fastain):
    records = list(SeqIO.parse(fastain,"fasta"))
    id_seqs=[]
    for i in range(len(records)):
        id=str(records[i].id)
        ss=str(records[i].seq)
        id_seqs.append([id,ss])
    
    id_seqs=pd.DataFrame(id_seqs,columns=["ID","Seq"])
    
    print("Read fasta file DONE!")
    return id_seqs

def Emb_BERT_bfd(data_pd):
    FeatureName=["ABa_bertF"+str(i+1) for i in range(1024)]
    sequences=data_pd["Seq"]
    bio_trans = BioTransformers(backend="protbert_bfd")
    embeddings = bio_trans.compute_embeddings(sequences, pool_mode=('cls','mean'),batch_size=2)

     
    mean_emb = embeddings['mean']
     
    ABaf=pd.DataFrame(mean_emb)
    ABaf.columns=FeatureName
    ABaf=pd.concat([data_pd,ABaf],axis=1)
    print(ABaf.shape)
    print("SEQUENCES embbeding DONE!")
    
    return ABaf
    
    
    





def predict(data_pd,std="./ML_Models/BTstd.joblib",model="./ML_Models/BertThermo.LR.joblib"):
    bestFeat=pd.read_csv("./ML_Models/FeatName1024.csv",header=0)
    col_name=bestFeat.columns
    
    data=data_pd
     
    std=joblib.load(std)
    model=joblib.load(model)
     
    
    X=data[col_name]
    X=std.transform(X)
    #print(X.shape)
    Xt=X[:,:90]
    #print(Xt.shape)
    #print(Xt)
    y_pred=model.predict(Xt)
    y_pred_prob=model.predict_proba(Xt)
    
    y_pred_label=[]
    for i in range(len(y_pred)):
        if y_pred[i]==0:
           yy=(round(1-y_pred_prob[i,1],3))*100
           yy= str(yy)+"%"
           y_pred_label.append(["NO",yy])
        else:
           yy=(round(y_pred_prob[i,1],3))*100
           yy= str(yy)+"%"
           y_pred_label.append(["YES",yy])
    
    y_pred_label=pd.DataFrame(y_pred_label,columns=["Thermophilic_Protein","Probability"])
    

    output=pd.concat([data_pd.iloc[:,:2],y_pred_label],axis=1)
    
    print("Prediction DONE!")
    
    return output



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Input a Fasta file, output the predction results in CSV')
 
    
    parser.add_argument('--infasta', type=str, default="./Data/test.fasta")
    parser.add_argument('--out', type=str, default="Results.Pred.csv")
 
    
    args = parser.parse_args()
    print(args)
    inFasta = args.infasta
    out = args.out+".res.pred.csv"
    fasta_pd=read_fasta(inFasta)
    data_pd=Emb_BERT_bfd(fasta_pd)
    output=predict(data_pd)
    output.to_csv(out)
    print("Plese see the predction results in ", out)
 
