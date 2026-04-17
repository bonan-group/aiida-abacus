from aiida import orm
from aiida.common import AttributeDict

from aiida_abacus.common import RelaxType
from aiida_abacus.common.opthold import RelaxOptions


def test_relax_options_aiida_dict_serializes_enum_to_string(aiida_profile_clean):
    options = RelaxOptions(relax_type=RelaxType.POSITIONS)
    data = options.aiida_dict().get_dict()

    assert data["relax_type"] == RelaxType.POSITIONS.value
