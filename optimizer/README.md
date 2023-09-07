# How to reproduce the experiments
The optimizer has been configured as a greedy algorithm which tries to satisfy given constraints.

### Dependencies

**Recommended: Python 3.7**

Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install all the dependencies by running the following commands.
```
pip install -r requirements.txt
```

## Execute the code
You can run the optimizer by running the following command:
```
python optimize_routing.py
python optimize_scheduling.py
python optimize_scheduling_routing.py
python link_failure.py
python network_upgrade.py
```