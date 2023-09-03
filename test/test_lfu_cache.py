import pytest
from chaserner.utils import LFUCache

def test_basic_operations():
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1
    cache.put(3, 3)  # should evict key 2
    assert cache.get(2) == None
    assert cache.get(3) == 3

def test_eviction_order():
    cache = LFUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    cache.get(1)
    cache.get(2)
    cache.get(2)
    cache.put(4, 4)  # should evict key 3 as it's the least frequently used
    assert cache.get(1) == 1
    assert cache.get(2) == 2
    assert cache.get(3) == None
    assert cache.get(4) == 4

def test_eviction_order_with_same_frequency():
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.get(1)
    cache.get(2)
    cache.put(3, 3)  # should evict key 1 as it's older than key 2
    assert cache.get(1) == None
    assert cache.get(2) == 2
    assert cache.get(3) == 3

def test_overwrite_value():
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(1, 10)  # should update the value for key 1
    assert cache.get(1) == 10

def test_eviction_after_multiple_accesses():
    cache = LFUCache(3)
    cache.put(1, 1)
    cache.put(2, 2)
    cache.put(3, 3)
    cache.get(1)
    cache.get(1)
    cache.get(2)
    cache.get(3)
    cache.get(3)
    cache.get(3)
    cache.put(4, 4)  # should evict key 2 as it's the least frequently used
    assert cache.get(1) == 1
    assert cache.get(2) == None
    assert cache.get(3) == 3
    assert cache.get(4) == 4

@pytest.mark.parametrize("ops, args, expected", [
    ([LFUCache.put, LFUCache.put, LFUCache.get, LFUCache.put, LFUCache.get, LFUCache.get, LFUCache.get],
     [(1, 1), (2, 2), (1,), (3, 3), (2,), (3,), (4,)],
     [None, None, 1, None, None, 3, None])
])
def test_parameterized(ops, args, expected):
    cache = LFUCache(2)
    for op, arg, exp in zip(ops, args, expected):
        assert op(cache, *arg) == exp
