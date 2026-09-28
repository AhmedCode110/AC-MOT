"""MAC-COMPAT runner (integration/compatibility only; method unchanged).
- puts the bundled YOLOX on sys.path;
- for --no_reid runs, stubs the ReID packages that tracker/embedding.py imports
  at module level but never calls when embeddings are disabled."""
import sys, types
sys.path.insert(0, "external/YOLOX")
if "--no_reid" in sys.argv:
    sys.modules.setdefault("torchreid", types.ModuleType("torchreid"))
    fr = types.ModuleType("external.adaptors.fastreid_adaptor")
    class FastReID:  # never instantiated with --no_reid
        def __init__(self, *a, **k):
            raise RuntimeError("ReID disabled")
    fr.FastReID = FastReID
    sys.modules["external.adaptors.fastreid_adaptor"] = fr
if __name__ == "__main__":  # macOS spawn-safe DataLoader workers
    import main
    main.main()
