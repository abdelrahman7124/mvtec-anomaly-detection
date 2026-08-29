"""Walk the dataset once and write reports/tables/manifest.csv."""

import _bootstrap  # noqa: F401
import json

from mvtec_eda.config import TABLES_DIR, ensure_output_dirs, get_data_root
from mvtec_eda.manifest import audit_manifest, build_manifest


def main() -> None:
    ensure_output_dirs()
    root = get_data_root()
    print(f"Dataset root: {root}")

    df = build_manifest(root)
    out = TABLES_DIR / "manifest.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows -> {out}")

    audit = audit_manifest(df)
    (TABLES_DIR / "integrity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nIntegrity audit:")
    for k, v in audit.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
