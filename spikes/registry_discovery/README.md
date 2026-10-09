# registry_discovery (THROWAWAY)

Run: `python spikes/registry_discovery/demo.py`

Creates a temp package with one environment module, runs `discover()`, then adds a second module
file and runs `discover()` again. Nothing else is edited.

Result:

```
before adding file: ['cartpole']
after adding pendulum.py: ['cartpole', 'pendulum']
```

Conclusion: one file with the decorator is enough; pkgutil discovery needs no central list.
