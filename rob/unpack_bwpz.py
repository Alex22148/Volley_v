from lib_pack import unpack_webp_pro

# #archive = r"F:\VolleyBall_2025\Data _from_TDS\BIN_WEBP\20251125_190822_1BIN\Raw_sets\CENTER_L\CENTER_L_0000.bwpz"
# archive = r"E:\sessions\session_20260419_111033\webp\session_20260419_111033\Raw_sets\CENTER_L\CENTER_L_0000.bwpz"
# #output_folder = r"F:\VolleyBall_2025\Data _from_TDS\BIN_WEBP\20251125_190822_1BIN\Raw_sets\CENTER_L\CENTER_L_0000x"
# output_folder = r"E:\sessions\session_20260419_111033\BIN_WEBP\session_20260419_111033\Raw_sets\CENTER_L\CENTER_L_000x"
#
#
# PASSWORD = "MojeBardzoTajneHaslo123"
#
# out_folder = unpack_webp_pro(
#             archive_path=archive,
#             output_folder=output_folder, #None,  # <archive>.folder
#             password=PASSWORD,
#             verify_checksums=True
#         )

from pathlib import Path

# import / definicja Twojej funkcji musi być dostępna
# from twoj_modul import unpack_webp_pro


PASSWORD = "MojeBardzoTajneHaslo123"


CAMERA_NAMES = [
    "CENTER_L",
    "CENTER_R",
    "RIGHT",
    "LEFT",
]


def unpack_session_bwpz_to_webp(
    session_root: str | Path,
    password: str,
    camera_names: list[str] | None = None,
    verify_checksums: bool = True,
    overwrite: bool = False,
):
    """
    Globalne rozpakowywanie archiwów .bwpz dla jednej sesji.

    Input:
        session_root:
            np. E:\\sessions\\session_20260419_111033

    Oczekiwana struktura wejściowa:
        <session_root>\\webp\\<session_name>\\Raw_sets\\<CAMERA>\\*.bwpz

    Struktura wyjściowa:
        <session_root>\\BIN_WEBP\\<session_name>\\Raw_sets\\<CAMERA>\\<archive_stem>

    Przykład:
        E:\\sessions\\session_20260419_111033\\webp\\session_20260419_111033\\Raw_sets\\CENTER_L\\CENTER_L_0000.bwpz

    zostanie rozpakowane do:
        E:\\sessions\\session_20260419_111033\\BIN_WEBP\\session_20260419_111033\\Raw_sets\\CENTER_L\\CENTER_L_0000
    """

    session_root = Path(session_root)
    session_name = session_root.name

    if camera_names is None:
        camera_names = CAMERA_NAMES

    input_raw_sets = session_root / "webp" / session_name / "Raw_sets"
    output_raw_sets = session_root / "BIN_WEBP" / session_name / "Raw_sets"

    if not input_raw_sets.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu wejściowego Raw_sets:\n{input_raw_sets}"
        )

    print("=" * 80)
    print("[BWPZ UNPACKER] Start")
    print(f"[SESSION ROOT] {session_root}")
    print(f"[INPUT ] {input_raw_sets}")
    print(f"[OUTPUT] {output_raw_sets}")
    print("=" * 80)

    unpacked = []
    skipped = []
    errors = []

    for camera_name in camera_names:
        camera_input_dir = input_raw_sets / camera_name
        camera_output_dir = output_raw_sets / camera_name

        if not camera_input_dir.exists():
            print(f"[WARN] Brak folderu kamery: {camera_input_dir}")
            skipped.append((camera_name, "missing_camera_folder"))
            continue

        archives = sorted(camera_input_dir.glob("*.bwpz"))

        if not archives:
            print(f"[WARN] Brak archiwów .bwpz w: {camera_input_dir}")
            skipped.append((camera_name, "no_archives"))
            continue

        print()
        print(f"[CAMERA] {camera_name}")
        print(f"[FOUND ] {len(archives)} archiwów")

        for archive_path in archives:
            output_folder = camera_output_dir / archive_path.stem

            if output_folder.exists() and not overwrite:
                print(f"[SKIP] Istnieje już: {output_folder}")
                skipped.append((archive_path, "output_exists"))
                continue

            try:
                print(f"[UNPACK] {archive_path}")
                print(f"   ->   {output_folder}")

                out_folder = unpack_webp_pro(
                    archive_path=str(archive_path),
                    output_folder=str(output_folder),
                    password=password,
                    verify_checksums=verify_checksums,
                )

                unpacked.append((archive_path, out_folder))
                print(f"[OK] {out_folder}")

            except Exception as e:
                print(f"[ERROR] Nie udało się rozpakować: {archive_path}")
                print(f"        {type(e).__name__}: {e}")
                errors.append((archive_path, e))

    print()
    print("=" * 80)
    print("[SUMMARY]")
    print(f"Rozpakowano: {len(unpacked)}")
    print(f"Pominięto:    {len(skipped)}")
    print(f"Błędy:        {len(errors)}")
    print("=" * 80)

    return {
        "session_root": session_root,
        "input_raw_sets": input_raw_sets,
        "output_raw_sets": output_raw_sets,
        "unpacked": unpacked,
        "skipped": skipped,
        "errors": errors,
    }


if __name__ == "__main__":
    session_root = r"E:\sessions\session_20260418_191949"

    result = unpack_session_bwpz_to_webp(
        session_root=session_root,
        password=PASSWORD,
        camera_names=CAMERA_NAMES,
        verify_checksums=True,
        overwrite=False,
    )

