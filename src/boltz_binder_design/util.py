import hashlib
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.tree_util as jtu


def fold_in(key: jax.dtypes.prng_key, name: str) -> jax.dtypes.prng_key:
    # hash name to int
    h = hashlib.sha256(name.encode("utf-8")).digest()
    h = int.from_bytes(h[-8:], "big")
    return jax.random.fold_in(key, h)


@dataclass(frozen=True, slots=True)
class _At:
    path: list[object]
    pytree: object
    cur_val: object  # we track this only so we can distinguish between DictKeys and SequenceKeys, seems a bit silly

    def _accessor(self, node):
        for key in self.path:
            match key:
                case jtu.DictKey(key):
                    node = node[key]
                case jtu.GetAttrKey(key):
                    node = getattr(node, key)
                case jtu.SequenceKey(key):
                    node = node[key]

        return node

    def __getattr__(self, key) -> "_At":
        return _At(
            self.path + [jtu.GetAttrKey(key)], self.pytree, getattr(self.cur_val, key)
        )

    def __getitem__(self, key) -> "_At":
        if isinstance(self.cur_val, dict):
            return _At(self.path + [jtu.DictKey(key)], self.pytree, self.cur_val[key])
        else:
            return _At(
                self.path + [jtu.SequenceKey(key)], self.pytree, self.cur_val[key]
            )

    def __call__(self, new_value):
        return eqx.tree_at(self._accessor, self.pytree, new_value)

    def replace(self, replace_fn: callable):
        return eqx.tree_at(self._accessor, self.pytree, replace_fn=replace_fn)


def At(pytree: object) -> _At:
    return _At([], pytree, pytree)
