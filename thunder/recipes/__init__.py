from thunder.recipes.base import BaseRecipe
from thunder.recipes.hf_transformers import HFTransformers


names_to_recipes = {
  "default": BaseRecipe,
  "hf-transformers": HFTransformers,
}


def get_recipe(name):
    return names_to_recipes.get(name)

def get_recipe_names():
    return list(names_to_recipes.keys())

def register_recipe(name, cls):
    names_to_recipes[name] = cls
