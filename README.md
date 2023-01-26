# BertThermo
Please install Anaconda first and then create a conda env.<br>
Follow by the commands as below.<br>
conda create -n BertThermo python=3.7<br>
conda activate BertThermo<br>
pip install -r requirements.txt<br>

<br> The env is bulit.
<br> run python predict.py -h and will see the help <br>
Input a Fasta file, output the predction results in CSV <br>

optional arguments: <br>
  -h, --help         show this help message and exit <br>
  --infasta INFASTA <br>
  --out OUT<br>
  
  for example <br>
  
python predict.py --infasta ./Data/Test.fasta --out Test.Result <br>
