# games
Agents play bargaining games on graphs. Reproduction of the results in "Persistence of discrimination: Revisiting Axtell, Epstein and Young" and "Lattice Dynamics of Inequity".

To install the requirements inside an environment, run
```
python3 -m venv env
source env/bin/activate
python3 -m  pip install -r requirements.txt
```

The Python code was developed and tested on Pyton 3.9.0.

To run a bargain game on a graph,
```
cd python/games
python3 run.py
```


#### yapf:
To format the python code, install [yapf](https://github.com/google/yapf) and run,
```
cd python
yapf -ir .
```


#### authors:

George Arampatzis (garampat at ethz.ch)

Pantelis Vlachas (pvlachas at ethz.ch)

at CSElab, ETH, Zurich, 2020