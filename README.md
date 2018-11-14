# Circuit Caching

Circuit caching features can be enabled declaratively or programmatically.

## Declarative

Circuit caching can be enabled through the dictionaries, CLI, or UI by setting the following arguments in the problem section:

```
'problem': {'name': 'energy',
            'circuit_caching': True,
            'caching_naughty_mode': True,
            'circuit_cache_file': 'myfile.pickle',
            'persist_cache': True
            },
```

Note that these arguments are optional, and settings which are left empty are disabled by default. The arguments behave as follows:

- `circuit_caching` - Enables standard caching. When any circuit(s) is run via the run_circuits module and caching is enabled, we check whether we have a saved Qobj in the cache module. If we do, we will deepcopy the cached Qobj and attempt to reparameterize it with the parameters in our uncompiled circuit(s). If this fails, the circuit(s) is not isomorphic with the Qobj, and we clear the cache, recompile the circuit(s) from scratch, and save the resulting Qobj to the cache. When saving a Qobj to the cache, we encode its structure into a mapping array to speed up the reparameterization later.         
- `caching_naughty_mode` -  Naughty mode takes some risks to speed up the calculation. It repeatedly reuses the original Qobj saved in the cache for execution to avoid deepcopying, and skips Qobj validation. 
- `circuit_cache_file`- If this filename is set and caching is enabled, the cache module will attempt to load cached Qobjs from this file, and will save, via pickle, the cached Qobjs to the file.
- `persist_cache`- If this is `False`, the cache will be cleared each time `QuantumAlgorithm` is instantiated. If set to `True`, the cached Qobjs will be reused across many Aqua runs. This is useful in cases where the user is rerunning a similar calculation many times, such as a dissociation curve. If the cache is being saved to a file, then this is simply saving the time of deserializing the pickled cache file. 

## Programmatic

Cache settings can be modified programmatically via the instructions below. Note that these are all disabled by default.
```
from qiskit_aqua.utils import circuit_cache

circuit_cache.use_caching = True
circuit_cache.naughty_mode = True
circuit_cache.cache_file = 'my_file'
circuit_cache.persist_cache = True
```