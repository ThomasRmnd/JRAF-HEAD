import argparse

import numpy as np

from jrafhead.config import setup_style
from jrafhead.loader import load_li9he8_shape

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Input path")
args = parser.parse_args()

setup_style()

method_optimization        = "li9he8_shape_muon__changing_veto__analysis__cdwpttchi2_1_55m_1_13s__omilrec_jvertex"
method_subtraction         = "li9he8_shape_muon__standard__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"
method_subtraction_neutron = "li9he8_shape_muon__with_neutron__analysis__cdwpttchi2_3m_1_2s__omilrec_jvertex"

data_optimization          = load_li9he8_shape(args.input, method_optimization)
data_subtraction           = load_li9he8_shape(args.input, method_subtraction)
data_subtraction_neutron   = load_li9he8_shape(args.input, method_subtraction_neutron)


def build_event_keys(d, decimals: int = 6) -> set[tuple]:
    keys = set()
    for i in range(len(d.run_id)):
        key = (
            int(d.run_id[i]),
            round(float(d.e_p[i]), decimals),
            round(float(d.e_d[i]), decimals),
            tuple(round(float(x), decimals) for x in d.pos_p_mm[i]),
            tuple(round(float(x), decimals) for x in d.pos_d_mm[i]),
        )
        keys.add(key)
    return keys


keys_opti = build_event_keys(data_optimization.signal)
keys_sub  = build_event_keys(data_subtraction.signal)
keys_subn = build_event_keys(data_subtraction_neutron.signal)

n_opti, n_sub, n_subn = len(keys_opti), len(keys_sub), len(keys_subn)

print(f"Optimization signal:         {n_opti} events")
print(f"Subtraction signal:          {n_sub} events")
print(f"Subtraction (neutron) signal:{n_subn} events")
print()

for name_a, keys_a, name_b, keys_b in [
    ("optimization", keys_opti, "subtraction",         keys_sub),
    ("optimization", keys_opti, "subtraction (neutron)", keys_subn),
    ("subtraction",  keys_sub,  "subtraction (neutron)", keys_subn),
]:
    common    = keys_a & keys_b
    only_a    = keys_a - keys_b
    only_b    = keys_b - keys_a
    print(f"{name_a} vs {name_b}:")
    print(f"  common:        {len(common)}")
    print(f"  only in {name_a}: {len(only_a)}")
    print(f"  only in {name_b}: {len(only_b)}")
    print(f"  Jaccard index: {len(common) / len(keys_a | keys_b):.4f}")
    print()

all_three   = keys_opti & keys_sub & keys_subn
only_opti   = keys_opti - keys_sub - keys_subn
only_sub    = keys_sub  - keys_opti - keys_subn
only_subn   = keys_subn - keys_opti - keys_sub
opti_sub    = (keys_opti & keys_sub) - keys_subn
opti_subn   = (keys_opti & keys_subn) - keys_sub
sub_subn    = (keys_sub  & keys_subn) - keys_opti

print("Three-way breakdown:")
print(f"  In all three:                        {len(all_three)}")
print(f"  Only optimization:                   {len(only_opti)}")
print(f"  Only subtraction:                    {len(only_sub)}")
print(f"  Only subtraction (neutron):          {len(only_subn)}")
print(f"  optimization & subtraction (not neutron):        {len(opti_sub)}")
print(f"  optimization & subtraction (neutron) (not sub):  {len(opti_subn)}")
print(f"  subtraction & subtraction (neutron) (not opti):  {len(sub_subn)}")