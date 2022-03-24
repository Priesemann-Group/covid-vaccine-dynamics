# Casusal Covid

## Installation

Clone with 
```bash
git clone --recurse-submodules git@github.com:Priesemann-Group/causal_covid.git
```
### Notes:
You need `python>3.8`.

Before you can use the code and rerun the analyses you have to:

- init the submodules:
	```bash
	#Init
	git submodule init
	# Update package manual (inside covid19_inference folder)
	cd covid19_inference
	git pull origin master
	```

- install the requirements with
	```bash
	pip install -r requirments.txt
	```
## Getting started

### Scenario calculation

You can use one of the inferred models (saved in ```./data/traces/```) to investigate what would happen in a 
counterfactual scenario with an alternative vaccination policy. 
Go into ```./scripts/``` and run ```calculate_scenario.py``` (```./scripts``` has to be the current
working directory). In ```calculate_scenario.py``` the path for the U2 and u3 matrices
of the alternative scenario have to be set.

### Inference

To infer the base reproduction number that is necessary for the scenario calculations
afterwards, go into ```./scripts/``` and run ```infer_single_age_group.py -i 1``` (for
the single dimensional models). Parameters are set inside the script. 

