from _lib_3D_estimate import (estiamte3D_work, configuration_est3D,
                              load_calibration, load_cam_trensforms)

import json
import numpy as np


def load_data_from_json(json_path: str, as_numpy=True):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    configuration = data["configuration"]
    P_all = np.array(data["P_all"], dtype=float) if as_numpy else data["P_all"]
    K_all = np.array(data["K_all"], dtype=float) if as_numpy else data["K_all"]
    dist_all = np.array(data["dist_all"], dtype=float) if as_numpy else data["dist_all"]
    C_all = np.array(data["C_all"], dtype=float) if as_numpy else data["C_all"]
    pair_transforms = data["pair_transforms"]
    conf_bat = np.array(data["conf_bat"], dtype=float) if as_numpy else data["conf_bat"]
    pts_bat = np.array(data["pts_bat"], dtype=float) if as_numpy else data["pts_bat"]

    return (
        configuration,
        P_all,
        K_all,
        dist_all,
        C_all,
        pair_transforms,
        conf_bat,
        pts_bat,
    )


# -------------------------

(
    configuration,
    P_all,
    K_all,
    dist_all,
    C_all,
    pair_transforms,
    conf_bat,
    pts_bat,
) = load_data_from_json("dane_testowe_3D_fixed.json", as_numpy=True)

selection_3D, analyze_3D, info_time = estiamte3D_work(pts_bat, configuration, P_all, K_all, dist_all, C_all, pair_transforms, work=False, conf_bat=conf_bat)


print("\nProcessing times:")
for name, time in info_time.items():
    print(f"- {name:13}: {time:7.3f} [ms]")

print('')
print(selection_3D)

''''
Processing times:
- GPU --> CPU  :   0.000 [ms]
- undistort    :   0.209 [ms]
- get3D        :   3.010 [ms]
- analyze_3D   :   3.021 [ms]
- selection_3D :   0.093 [ms]

[[-9.001736640930176, -4.500884532928467, 0.19389407336711884], [-9.031225204467773, 4.47531270980835, 0.19931767880916595], [-3.1154768466949463, -4.445175647735596, 0.17773811519145966], [-3.0574262142181396, -0.015115827322006226, 0.6652971506118774], [2.9314799308776855, -0.008460324257612228, 0.3747028112411499], [2.955422878265381, 4.511605262756348, 0.16925497353076935], [4.297738552093506, -4.489407062530518, 0.5444931387901306], [9.03358268737793, -4.507431983947754, 0.22287586331367493], [9.053871154785156, 4.51397180557251, 0.21355721354484558], [-2.9387948513031006, 4.551052570343018, 3.6411406993865967], [3.1636106967926025, -4.523082256317139, 3.6469082832336426], [3.176490306854248, -0.050850287079811096, 3.64210844039917]]

'''