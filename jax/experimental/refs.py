import jax

from jax import nn
from jax import numpy as jnp
from jax import random
from jax.util import partial

from collections import defaultdict
import threading

class NoValue:
  pass

no_value = NoValue()

class Ref(threading.local):
  """A container for managing state in Python."""
  __slots__ = ["value"]

  def __init__(self, value=no_value):
    super().__init__()
    self.value = value

  def __repr__(self):
    return self.__class__.__name__ + '(' + repr(self.value) + ')'

  def load(self):
    if self.value is no_value:
      raise RuntimeError("Cannot load from empty ref.")
    return self.value

  def store(self, value):
    self.value = value

  # def swap(self, value):
  #   if self.value is no_value:
  #     raise RuntimeError("Cannot swap into empty ref.")
  #   value, self.value = self.value, value
  #   return value

  # def initialize(self, value)
  #   if self.value is no_value:
  #     self.value = value
  #   return self.value

# Three choices for Ref's pytree behavior:
# 1. register Ref as a pytree with its object identity as metadata
# 2. register Ref as a pytree that always throws an error
# 3. (chosen) don't register Ref as a pytree (treat it as a leaf)
#    --> enables using Refs in tree_util but not jit

def tree_load(ref_tree, typ=Ref):
  loaded = set()
  def load(ref):
    if isinstance(ref, typ) and ref not in loaded:
      loaded.add(ref)
      return ref.load()
  return jax.tree_map(load, ref_tree)

def tree_store(ref_tree, val_tree, typ=Ref):
  stored = set()
  def store(ref, val):
    if isinstance(ref, typ) and ref not in stored:
      stored.add(ref)
      ref.store(val)
  jax.tree_multimap(store, ref_tree, val_tree)

def collect(fun, ref_tree):
  def inner(*args, **kwargs):
    out = fun(*args, **kwargs)
    val_tree = tree_load(ref_tree)
    return val_tree, out
  return inner

def inject(fun, ref_tree):
  def inner(val_tree, *args, **kwargs):
    tree_store(ref_tree, val_tree)
    return fun(*args, **kwargs)
  return inner

# maybe a function that does both

# examples

refs = [Ref(), Ref()]

# foo :: [Writer(refs)] float -> [Writer(refs)] float
def foo(x):
  y = x ** 2
  refs[0].store(y)
  refs[1].store(y - 1)
  return y + 1

# pure_foo :: float -> List[float], float
pure_foo = collect(foo, refs)
assert pure_foo(3) == ([9, 8], 10)

aux = Ref()
def fn_with_aux(x):
  y = x ** 2
  aux.store(y)
  z = y + 1
  return z

assert collect(fn_with_aux, aux)(2) == (4, 5)
# with_refs(grad, yref)(bar)

## tagging

_tag_refs = Ref(dict())
def tag(name, value=no_value):
  _tag_vals = _tag_refs.load()
  if name not in _tag_vals:
    assert value is not no_value
    _tag_vals[name] = value
  else:
    value, _tag_vals[name] = _tag_vals[name], value
  _tag_refs.store(_tag_vals)
  return value

# _tag_refs = defaultdict(lambda: Ref(no_value))
# def tag(name, value=no_value):
#   ref = _tag_refs[name]
#   if ref.value is no_value:
#     ref.store(value)
#   else:
#     return ref.load()

def tagged_fn_with_aux(x):
  y = x ** 2
  tag('y', y)
  z = y + 1
  return z

assert collect(tagged_fn_with_aux, _tag_refs)(2) == ({'y': 4}, 5)

def tagged_fn_with_intermediate(x):
  y = x ** 2
  y = tag('y', y)
  z = y + 1
  return z

assert inject(tagged_fn_with_intermediate, _tag_refs)({'y': 3}, 2) == 4

## PRNGs

_global_PRNG_key = Ref()
def next_key():
  key1, key2 = random.split(_global_PRNG_key.load())
  _global_PRNG_key.store(key1)
  return key2

# example

def stateful_normal():
  return random.normal(next_key())

pure_normal = inject(stateful_normal, _global_PRNG_key)
pure_normal(random.PRNGKey(2))

# # wrapping transforms

# @lu.transformation
# def _collect_and_inject(fun, ref_tree, val_tree, *args, **kwargs):
#   tree_store(ref_tree, val_tree)
#   out = yield args, kwargs
#   val_tree = tree_load(ref_tree)
#   return val_tree, out

# _treelike_kwargs = {'in_axes', 'out_axes', 'tangents'}
# _listlike_kwargs = {'static_argnums'}

# def with_refs(transform, ref_tree, **ref_kwargs):
#   def transform_with_refs(fun, *transform_args, **transform_kwargs):
#     fun = collect_and_inject(fun, )
#     def inner(*args, **kwargs):
#       fun = lu.wrap_init(fun)
#       fun = _collect_and_inject(fun)
#       args, in_tree = tree_flatten((args, kwargs))
#       flat_fun, out_tree = flatten_fun(f, in_tree)
#     for k, v in transform_kwargs.items():
#       if k in _listlike_kwargs:

#     def inner(*args, **kwargs):
#       for k, v in transform_kwargs.items():
#         if k in _listlike_kwargs:
#           kwargs
#         args_flat, in_tree = tree_flatten((tree_load(refs), *args))
#       fun = collect_and_inject(fun, refs)

# neural nets

class Parameter(Ref):
  """A trainable parameter."""

class Buffer(Ref):
  """A container for non-trainable state."""


class _ModuleMeta(type):
  def __init__(cls, name, bases, attrs):
    super(_ModuleMeta, cls).__init__(name, bases, attrs)
    def from_kv(keys, values):
      module = cls.__new__(cls)
      module.__dict__.update(**dict(zip(keys, values)))
      return module
    jax.tree_util.register_pytree_node(
        cls,
        lambda m: (list(m.__dict__.values()), list(m.__dict__.keys())),
        from_kv)

class Module(metaclass=_ModuleMeta):
  def __repr__(self):
    s = ', '.join(k + '=' + repr(v) for k, v in self.__dict__.items())
    return self.__class__.__name__ + '(' + s + ')'
  # def variables(self, typ=Ref):
  #   def inner(v):
  #     if isinstance(v, Module):
  #       yield from v.variables(typ)
  #     else:
  #       yield v
  #   for v in self.__dict__.values():
  #     if isinstance(v, Module):
  #       yield from v.variables(typ)
  #     else:
  #       for sub in jax.tree_flatten(v)[0]:
  #         yield from inner(sub)


class Linear(Module):
  def __init__(self, nI, nO, bias=True,
               weight_init=nn.initializers.lecun_normal(),
               bias_init=nn.initializers.zeros):
    self.W = Parameter(weight_init(next_key(), (nI, nO)))
    if bias:
      self.b = Parameter(bias_init(next_key(), (nO,)))
  def __call__(self, x):
    return x @ self.W.load() + self.b.load()

class Sequential(Module):
  def __init__(self, layers):
    self.layers = layers
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

def arch1():
  layer1 = Linear(2, 2)
  layer2 = Linear(2, 2)
  layer1.W = layer2.W
  return Sequential([layer1, layer2])

def arch2():
  layer1 = Linear(2, 2)
  layer2 = layer1
  return Sequential([layer1, layer2])

model1 = inject(arch1, _global_PRNG_key)(random.PRNGKey(1))
print('model1', model1)
model2 = inject(arch2, _global_PRNG_key)(random.PRNGKey(1))
print('model2', model2)

def loss1(x):
  return jnp.sum(model1(x))

def loss2(x):
  return jnp.sum(model2(x))

params1 = tree_load(model1, Parameter)
print('params1', params1)
params2 = tree_load(model2, Parameter)
print('params2', params2)

def train_step1(params, x):
  grads = jax.grad(inject(loss1, model1))(params, x)
  print('grads1', grads)
  return jax.tree_multimap(lambda p, g: p - g, params, grads)

def train_step2(params, x):
  grads = jax.grad(inject(loss2, model2))(params, x)
  print('grads2', grads)
  return jax.tree_multimap(lambda p, g: p - g, params, grads)

x = random.normal(random.PRNGKey(0), (2,))
params1 = train_step1(params1, x)
print('params1', params1)
tree_store(model1, params1)
print('model1', model1)
params2 = train_step2(params2, x)
print('params2', params2)
tree_store(model2, params2)
print('model2', model2)
