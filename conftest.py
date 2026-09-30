# Ensures the project root (and therefore the `waste_sorter` package) is
# importable regardless of how pytest is invoked (`pytest`, `python -m
# pytest`, from a subdirectory, etc.) — pytest's default "prepend" import
# mode adds the directory containing this file to sys.path.
