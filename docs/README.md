# Automatically Generating the Integrated Aqua, Aqua-Chemistry, Aqua Artificial Intelligence, Aqua Optimization, Aqua Finance, and Aqua Tutorials Documentation

1. Make sure you have `Sphinx` >= 1.7.6, `sphinxcontrib-fulltoc` >= 1.2.0, and `sphinxcontrib-websupport` >= 1.1.0 installed
   in the same Python environment where you have `aqua` and `aqua-chemistry` installed.
2. The `aqua` and `aqua-chemistry` repositories must be installed via `git clone` next to each other in the same folder of the
   file system.
3. From the `docs` folder of `aqua-chemistry`, issue the following commands:

   - `make clean`
   - `sphinx-apidoc -f -o . ..`
   - `make html`
   
4. Repeat steps 3.a, 3.b and 3.c from the `docs` folder of `aqua`
5. The Aqua, Aqua-Chemistry, Aqua Artificial Intelligence, Aqua Optimization, Aqua Finance, and Aqua Tutorials documentation 
will self-generate in the `_build/html` subfolder of the `docs` folder of `aqua`, with `index.html` being the main file
   
Skipping step 3 above will only generate the Aqua documentation, without the Aqua-Chemistry, Aqua Artificial Intelligence,
Aqua Optimization and Aqua Tutorial integrated into it.
