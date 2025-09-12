# inspired by geomstats; to be merged there soon

import importlib
import logging
import os
import sys
import types


def get_backend_name():
    return os.environ.get("GEOMSTATS_BACKEND", "numpy")


BACKEND_NAME = get_backend_name()


BACKEND_ATTRIBUTES = {
    "": [
        "geomspace",
        "scatter_sum_1d",
        "square",
        "argsort",
        "to_torch",
        "diag",
        "to_device",
        "pinv",
    ],
    "sparse": [
        "to_dense",
        "from_scipy_coo",
        "from_scipy_csc",
        "from_scipy_csr",
        "from_scipy_dia",
        "to_scipy_csc",
        "to_scipy_dia",
        "csr_matrix",
        "csc_matrix",
        "coo_matrix",
        "dia_matrix",
        "to_torch_csc",
        "to_torch_dia",
        "to_torch_coo",
        "to_coo",
        "to_csc",
        "to_csr",
    ],
}


class BackendImporter:
    """Importer class to create the backend module."""

    def __init__(self, path):
        self._path = self.name = path
        self.loader = self

    @staticmethod
    def _import_backend(backend_name):
        try:
            return importlib.import_module(f"geomfum._backend.{backend_name}")
        except ModuleNotFoundError:
            raise RuntimeError(f"Unknown backend '{backend_name}'")

    def _create_backend_module(self, backend_name):
        backend = self._import_backend(backend_name)

        new_module = types.ModuleType(self._path)
        new_module.__file__ = backend.__file__

        for module_name, attributes in BACKEND_ATTRIBUTES.items():
            if module_name:
                try:
                    submodule = getattr(backend, module_name)
                except AttributeError:
                    raise RuntimeError(
                        f"Backend '{backend_name}' exposes no '{module_name}' module"
                    ) from None
                new_submodule = types.ModuleType(f"{self._path}.{module_name}")
                new_submodule.__file__ = submodule.__file__
                setattr(new_module, module_name, new_submodule)
            else:
                submodule = backend
                new_submodule = new_module

            for attribute_name in attributes:
                try:
                    attribute = getattr(submodule, attribute_name)

                except AttributeError:
                    if module_name:
                        error = (
                            f"Module '{module_name}' of backend '{backend_name}' "
                            f"has no attribute '{attribute_name}'"
                        )
                    else:
                        error = (
                            f"Backend '{backend_name}' has no "
                            f"attribute '{attribute_name}'"
                        )

                    raise RuntimeError(error) from None
                else:
                    setattr(new_submodule, attribute_name, attribute)

        return new_module

    def find_module(self, fullname, path=None):
        """Find module."""
        if self._path != fullname:
            return None
        return self

    def load_module(self, fullname):
        """Load module."""
        if fullname in sys.modules:
            return sys.modules[fullname]

        module = self._create_backend_module(BACKEND_NAME)
        module.__name__ = f"geomfum.{BACKEND_NAME}"
        module.__loader__ = self
        sys.modules[fullname] = module

        logging.debug(f"geomfum is using {BACKEND_NAME} backend")
        return module

    def find_spec(self, fullname, path=None, target=None):
        """Find module."""
        return self.find_module(fullname, path=path)


sys.meta_path.append(BackendImporter("geomfum.backend"))
