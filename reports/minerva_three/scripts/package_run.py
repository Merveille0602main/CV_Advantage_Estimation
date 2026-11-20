"""Copy the audited low-dimensional study and all scored test snapshots to a release folder."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from verify import check


def main(run,destination):
    check(run)
    sources={}
    for task in (13,19,25):
        for anchor in (run/str(task)/"anchors").glob("*.json"):
            row=json.loads(anchor.read_text(encoding="utf8"))
            if row["split"]=="test":
                path=anchor.parent.parent/"snapshots"/(anchor.stem+".npz")
                sources[path.relative_to(run).as_posix()]=row["snapshot_sha256"]
    assert len(sources)==90
    digests={}
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        rel=path.relative_to(run)
        if any(part.startswith(".") or part=="logs" for part in rel.parts) or path.name=="checksums.json":
            continue
        key=rel.as_posix()
        if "snapshots" in rel.parts and key not in sources:
            continue
        if path.suffix not in (".npz",".pt",".csv",".json"):
            continue
        target=destination/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if key in sources:
            assert digest==sources[key]
        shutil.copy2(path,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        digests[key]=digest
    assert sources.keys()<=digests.keys()
    (destination/"checksums.json").write_text(json.dumps(digests,indent=2,sort_keys=True),encoding="utf8")
    print(len(digests),"files with SHA-256;",len(sources),"test snapshots")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    main(args.run,args.destination)
