#!/usr/bin/env python3
"""Export verified weights/metadata to R2, or restore a portable inference bundle."""
import argparse
import json
from pathlib import Path
from wavefront_denoiser.model_artifacts import build_bundle, upload_bundle, download_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--metadata', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--destination', required=True, help='r2://denoiser/model-runs/UNIQUE_ID')
    export.add_argument('--include-resume', action='store_true')
    export.add_argument('--results', type=Path)
    upload = sub.add_parser('upload')
    upload.add_argument('bundle', type=Path); upload.add_argument('destination')
    fetch = sub.add_parser('download')
    fetch.add_argument('source'); fetch.add_argument('output', type=Path)
    a = p.parse_args()
    if a.command == 'export':
        build_bundle(a.checkpoint, a.metadata, a.output, include_resume=a.include_resume, results=a.results)
        result = upload_bundle(a.output, a.destination)
        (a.output/'upload-receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    elif a.command == 'upload': result = upload_bundle(a.bundle, a.destination)
    else:
        manifest = download_bundle(a.source, a.output)
        result = {'source': a.source, 'output': str(a.output), 'verified': True, 'epoch': manifest['epoch']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
