from typing import Any, Type

from thunder.recipes.base import BaseRecipe
from thunder.recipes.hf_transformers import HFTransformers


names_to_recipes: dict[str : type[Any]] = {
    "default": BaseRecipe,
    "hf-transformers": HFTransformers,
}


def get_recipe_class(name: str) -> type[Any]:
    return names_to_recipes.get(name)


def get_recipes() -> list[str]:
    return list(names_to_recipes.keys())


def register_recipe(name: str, cls: type[Any]):
    if name == "auto":
        raise ValueError("Recipe name 'auto' is reserved.")
    names_to_recipes[name] = cls
