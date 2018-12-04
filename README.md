# SinglePerceptron
## syntax for running the perceptron:
* For running perceptron use either of the below methods
- python perceptron.py -i <inputfile.tsv> -o <outputfile.tsv>
- python perceptron.py --data <inputfile.tsv> --output <outputfile.tsv>

### you can also run perceptron.py without passing any files in input or output then you'll be asked to subsequently enter them in output prompt

### In gerneral the number of iterations is set to 100 however you can change them if you want by sending through command line using '-t' or "--iter"
* Example:
* python perceptron.py --data <inputfile.tsv> --output <outputfile.tsv> --iter <150>
 
* python perceptron.py -i <inputfile.tsv> -o <outputfile.tsv> -t <iterations>
