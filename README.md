# causal_covid


Installation with 
```bash
git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_soccer.git
```
## Notes:
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

- install the requirements
	```bash
	pip install -r requirments.txt
	```