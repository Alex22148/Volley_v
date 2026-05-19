#!/usr/bin/env python
"""
Skrypt do eksportu timestampów z sesji nagrania.

Użycie:
    python export_timestamps.py path/to/session_folder
    
Wygeneruje plik timestamps.json w katalogu sesji.
"""

import sys
from pathlib import Path

# Dodaj katalog główny do path
sys.path.insert(0, str(Path(__file__).parent))

from bin_export import export_session_timestamps


def main():
    # 🔹 WPISZ TUTAJ ŚCIEŻKĘ DO SESJI:
    session_dir = Path(r"E:\sessions\session_20260310_134522")
    
    # Lub użyj argumentu z linii poleceń:
    if len(sys.argv) >= 2:
        session_dir = Path(sys.argv[1])
    
    if not session_dir.exists():
        print(f"❌ Katalog nie istnieje: {session_dir}")
        sys.exit(1)
    
    if not session_dir.is_dir():
        print(f"❌ To nie jest katalog: {session_dir}")
        sys.exit(1)
    
    print(f"[EXPORT] Analizuję sesję: {session_dir}")
    print("=" * 60)
    
    try:
        result = export_session_timestamps(session_dir)
        
        print("\n" + "=" * 60)
        print("✅ GOTOWE!")
        print(f"\nPlik: {session_dir / 'timestamps.json'}")
        
        # Podsumowanie
        print(f"\nLiczba kamer: {len(result['cameras'])}")
        for role, timestamps in result['cameras'].items():
            print(f"  {role}: {len(timestamps)} klatek")
        
        # Analiza synchronizacji
        if result.get('sync_analysis', {}).get('first_frame_diff_ms'):
            print("\n🔍 ANALIZA SYNCHRONIZACJI:")
            diffs = result['sync_analysis']['first_frame_diff_ms']
            max_diff = max(abs(d) for d in diffs.values())
            
            if max_diff < 1.0:
                print(f"  ✅ Kamery zsynchronizowane (max różnica: {max_diff:.3f} ms)")
            elif max_diff < 10.0:
                print(f"  ⚠️  Niewielka desynchronizacja (max: {max_diff:.3f} ms)")
            else:
                print(f"  ❌ Duża desynchronizacja (max: {max_diff:.3f} ms)")
                print("     Kamery mogą nie mieć wspólnego zegara PTP!")
        
        if result.get('sync_analysis', {}).get('offset_stability'):
            print("\n[PTP] Stabilność offsetu (Master/Slave):")
            for role, stats in result['sync_analysis']['offset_stability'].items():
                status = "✅ STABILNY" if stats['is_stable'] else "⚠️ NIESTABILNY"
                print(f"  {role}: {status}")
                print(f"    średni offset: {stats['mean_offset_ms']:+.3f} ms")
                print(f"    odchylenie std: {stats['std_offset_ms']:.3f} ms")
                print(f"    max dryft: {stats['max_drift_ms']:.3f} ms")
        
    except Exception as e:
        print(f"\n❌ BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
