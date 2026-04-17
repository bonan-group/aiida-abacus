from aiida import orm

from aiida_abacus.common import recursive_merge


def test_recursive_merge_clones_existing_aiida_nodes():
    """Merged mappings should not alias nodes from the original inputs."""
    original_node = orm.Dict(dict={"ecutwfc": 60.0})
    left = {"abacus": {"parameters": original_node}}
    right = {"abacus": {"metadata": {"options": {"withmpi": False}}}}

    merged = recursive_merge(left, right)

    assert merged["abacus"]["parameters"] is not original_node
    assert merged["abacus"]["parameters"].get_dict() == original_node.get_dict()
    assert merged["abacus"]["metadata"]["options"]["withmpi"] is False
