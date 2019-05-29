# ROUGE scoring See et al. (2017)
Repository to replicate the ROUGE scores from [See et al. (2017)](https://nlp.stanford.edu/pubs/see2017get.pdf). 

We find that the reported scores correspond to those produced by the python re-implementation [py-rouge](https://pypi.org/project/py-rouge/), instead of those by produced by the official Rouge 155 Perl wrapper [pyrouge](https://pypi.org/project/pyrouge/). 

The [evaluate.py](evaluate.py) script accepts a 'hypothesis' folder and a 'reference' folder. The ROUGE scores computed with py-rouge and pyrouge respectively are then computed and printed to standard out.

The [test_output](test_output) folder, contains the test outputs from See et al. (2017), that can be downloaded from the README.md of the [official repository](https://github.com/abisee/pointer-generato://github.com/abisee/pointer-generator).

## Setup
`pip install py-rouge pyrouge`

### pyrouge prerequisites
Ensure Perl XML library is installed:  
On Arch Linux: `sudo pacman -S perl-xml-xpath`  
On Ubuntu: `sudo apt-get install libxml-parser-perl`

ROUGE 155 install tips/debugging:  
https://stackoverflow.com/questions/47045436/how-to-install-the-python-package-pyrouge-on-microsoft-windows

## Evaluate
Note that pyrouge evaluates ~4x as slow as py-rouge, so some patience is required.
```bash
# Evaluate Pointer Generator
$> python evaluate.py test_output/pointer-gen test_output/reference

Python (py-rouge) scores:
         ROUGE-1 (F1): 36.43
         ROUGE-2 (F1): 15.66
         ROUGE-L (F1): 33.42

Perl (pyrouge) scores:
         ROUGE-1 (F1): 36.16
         ROUGE-2 (F1): 15.61
         ROUGE-L (F1): 33.21

# Evaluate Pointer Generator + Coverage
$> python evaluate.py test_output/pointer-gen-cov test_output/reference

Python (py-rouge) scores:
         ROUGE-1 (F1): 39.53
         ROUGE-2 (F1): 17.28
         ROUGE-L (F1): 36.38

Perl (pyrouge) scores:
         ROUGE-1 (F1): 39.24
         ROUGE-2 (F1): 17.22
         ROUGE-L (F1): 36.15

```
