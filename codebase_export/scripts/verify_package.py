#!/usr/bin/env python3
"""
Standalone verification script for ensemble transcription packages.

This script verifies the integrity of transcription packages by validating
the run_manifest.json and recomputing SHA256 hashes for all artifacts.

Usage:
    python scripts/verify_package.py <package_path>
    python scripts/verify_package.py --manifest <manifest_path>
    python scripts/verify_package.py --batch <directory_with_packages>
"""

import os
import sys
import json
import argparse
import hashlib
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.manifest import verify_manifest_integrity, RunManifest
    from pydantic import ValidationError
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class PackageVerifier:
    """Verifies integrity of ensemble transcription packages"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {
            'packages_verified': 0,
            'packages_passed': 0,
            'packages_failed': 0,
            'total_artifacts': 0,
            'total_bytes': 0,
            'verification_errors': []
        }
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}] {level}: "
        print(f"{prefix}{message}")
    
    def verify_package(self, package_path: str) -> Tuple[bool, Dict[str, any]]:
        """
        Verify a complete transcription package.
        
        Args:
            package_path: Path to package (ZIP file or directory)
            
        Returns:
            Tuple of (verification_passed, verification_report)
        """
        package_path = Path(package_path)
        
        if not package_path.exists():
            return False, {"error": f"Package not found: {package_path}"}
        
        self.log(f"Verifying package: {package_path}")
        
        if package_path.suffix.lower() == '.zip':
            return self._verify_zip_package(package_path)
        elif package_path.is_dir():
            return self._verify_directory_package(package_path)
        else:
            return False, {"error": f"Unsupported package format: {package_path}"}
    
    def _verify_zip_package(self, zip_path: Path) -> Tuple[bool, Dict[str, any]]:
        """Verify a ZIP package by extracting and checking contents"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract ZIP file
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    zip_file.extractall(temp_path)
                
                # Find manifest file
                manifest_files = list(temp_path.rglob("run_manifest.json"))
                if not manifest_files:
                    return False, {"error": "No run_manifest.json found in package"}
                
                # Verify each manifest found
                overall_passed = True
                reports = []
                
                for manifest_path in manifest_files:
                    session_dir = manifest_path.parent
                    passed, report = self._verify_manifest_in_directory(session_dir)
                    reports.append(report)
                    if not passed:
                        overall_passed = False
                
                return overall_passed, {
                    "package_type": "zip",
                    "manifest_count": len(manifest_files),
                    "reports": reports
                }
                
        except Exception as e:
            return False, {"error": f"Failed to verify ZIP package: {e}"}
    
    def _verify_directory_package(self, dir_path: Path) -> Tuple[bool, Dict[str, any]]:
        """Verify a directory package"""
        # Look for manifest file
        manifest_path = dir_path / "run_manifest.json"
        if not manifest_path.exists():
            # Look for manifest in subdirectories
            manifest_files = list(dir_path.rglob("run_manifest.json"))
            if not manifest_files:
                return False, {"error": "No run_manifest.json found in directory"}
            
            # Verify all manifests found
            overall_passed = True
            reports = []
            
            for manifest_path in manifest_files:
                session_dir = manifest_path.parent
                passed, report = self._verify_manifest_in_directory(session_dir)
                reports.append(report)
                if not passed:
                    overall_passed = False
            
            return overall_passed, {
                "package_type": "directory",
                "manifest_count": len(manifest_files),
                "reports": reports
            }
        else:
            # Single manifest in root directory
            passed, report = self._verify_manifest_in_directory(dir_path)
            return passed, {
                "package_type": "directory",
                "manifest_count": 1,
                "reports": [report]
            }
    
    def _verify_manifest_in_directory(self, session_dir: Path) -> Tuple[bool, Dict[str, any]]:
        """Verify a manifest and its artifacts within a directory"""
        manifest_path = session_dir / "run_manifest.json"
        
        try:
            start_time = time.time()
            
            # Load and validate manifest schema
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            try:
                manifest = RunManifest(**manifest_data)
            except ValidationError as e:
                return False, {
                    "error": "Manifest schema validation failed",
                    "validation_errors": str(e)
                }
            
            # Use utility function to verify integrity
            passed, errors = verify_manifest_integrity(str(manifest_path), recompute_hashes=True)
            
            verification_time = time.time() - start_time
            
            # Generate detailed report
            report = {
                "manifest_path": str(manifest_path),
                "run_id": manifest.run_id,
                "session_id": manifest.session_id,
                "project_id": manifest.project_id,
                "validation_passed": passed,
                "verification_time_seconds": round(verification_time, 2),
                "artifact_count": len(manifest.artifacts),
                "total_bytes": manifest.total_bytes,
                "processing_duration": manifest.processing_duration_seconds,
                "manifest_version": manifest.manifest_version
            }
            
            if errors:
                report["validation_errors"] = errors
                report["error_count"] = len(errors)
            
            # Add artifact type breakdown
            artifact_types = {}
            for artifact in manifest.artifacts:
                artifact_type = artifact.artifact_type
                if artifact_type not in artifact_types:
                    artifact_types[artifact_type] = {"count": 0, "total_bytes": 0}
                artifact_types[artifact_type]["count"] += 1
                artifact_types[artifact_type]["total_bytes"] += artifact.bytes
            
            report["artifact_breakdown"] = artifact_types
            
            # Check for input media integrity
            if manifest.media_sha256 and manifest.media_path:
                media_path = session_dir / manifest.media_path
                if media_path.exists():
                    try:
                        computed_hash = self._compute_file_hash(media_path)
                        media_integrity_passed = computed_hash == manifest.media_sha256
                        report["input_media_integrity"] = {
                            "passed": media_integrity_passed,
                            "expected_sha256": manifest.media_sha256,
                            "computed_sha256": computed_hash,
                            "file_size": media_path.stat().st_size
                        }
                        if not media_integrity_passed:
                            passed = False
                    except Exception as e:
                        report["input_media_integrity"] = {
                            "passed": False,
                            "error": str(e)
                        }
                        passed = False
                else:
                    report["input_media_integrity"] = {
                        "passed": False,
                        "error": "Input media file not found"
                    }
                    passed = False
            
            # Update global statistics
            self.results['total_artifacts'] += len(manifest.artifacts)
            self.results['total_bytes'] += manifest.total_bytes
            
            if self.verbose:
                self.log(f"Manifest verification {'PASSED' if passed else 'FAILED'}: {manifest_path}")
                if errors:
                    for error in errors:
                        self.log(f"  - {error}", "ERROR")
            
            return passed, report
            
        except Exception as e:
            return False, {
                "error": f"Failed to verify manifest: {e}",
                "manifest_path": str(manifest_path)
            }
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def verify_batch(self, directory: str) -> Dict[str, any]:
        """
        Verify all packages in a directory.
        
        Args:
            directory: Directory containing packages to verify
            
        Returns:
            Batch verification report
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            return {"error": f"Directory not found: {directory}"}
        
        self.log(f"Starting batch verification in: {directory}")
        
        # Find all packages (ZIP files and directories with manifests)
        packages = []
        
        # Find ZIP files
        for zip_file in directory.glob("*.zip"):
            packages.append(zip_file)
        
        # Find directories with manifests
        for manifest_file in directory.rglob("run_manifest.json"):
            if manifest_file.parent not in [p.parent if p.suffix == '.zip' else p for p in packages]:
                packages.append(manifest_file.parent)
        
        if not packages:
            return {"error": "No packages found in directory"}
        
        self.log(f"Found {len(packages)} packages to verify")
        
        # Verify each package
        verification_results = []
        start_time = time.time()
        
        for package in packages:
            passed, report = self.verify_package(package)
            verification_results.append({
                "package_path": str(package),
                "verification_passed": passed,
                "report": report
            })
            
            self.results['packages_verified'] += 1
            if passed:
                self.results['packages_passed'] += 1
            else:
                self.results['packages_failed'] += 1
                self.results['verification_errors'].append({
                    "package": str(package),
                    "error": report.get('error', 'Validation failed')
                })
        
        batch_time = time.time() - start_time
        
        batch_report = {
            "batch_directory": str(directory),
            "packages_found": len(packages),
            "packages_verified": self.results['packages_verified'],
            "packages_passed": self.results['packages_passed'],
            "packages_failed": self.results['packages_failed'],
            "total_artifacts": self.results['total_artifacts'],
            "total_bytes": self.results['total_bytes'],
            "batch_verification_time": round(batch_time, 2),
            "verification_results": verification_results
        }
        
        if self.results['verification_errors']:
            batch_report["errors"] = self.results['verification_errors']
        
        return batch_report
    
    def print_summary(self):
        """Print verification summary"""
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        print(f"Packages verified: {self.results['packages_verified']}")
        print(f"Packages passed:   {self.results['packages_passed']}")
        print(f"Packages failed:   {self.results['packages_failed']}")
        print(f"Total artifacts:   {self.results['total_artifacts']}")
        print(f"Total bytes:       {self.results['total_bytes']:,}")
        
        if self.results['packages_verified'] > 0:
            success_rate = (self.results['packages_passed'] / self.results['packages_verified']) * 100
            print(f"Success rate:      {success_rate:.1f}%")
        
        if self.results['verification_errors']:
            print(f"\nErrors encountered:")
            for error in self.results['verification_errors']:
                print(f"  - {error['package']}: {error['error']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Verify integrity of ensemble transcription packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s package.zip                    # Verify a ZIP package
  %(prog)s /path/to/session/directory     # Verify a directory package  
  %(prog)s --manifest run_manifest.json  # Verify specific manifest
  %(prog)s --batch /path/to/packages      # Verify all packages in directory
  %(prog)s --json output.json package.zip # Save report as JSON
        """
    )
    
    parser.add_argument(
        'package_path',
        nargs='?',
        help='Path to package file/directory to verify'
    )
    
    parser.add_argument(
        '--manifest',
        help='Path to specific manifest file to verify'
    )
    
    parser.add_argument(
        '--batch',
        help='Directory containing multiple packages to verify'
    )
    
    parser.add_argument(
        '--json',
        help='Save verification report as JSON file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-hash-check',
        action='store_true',
        help='Skip hash recomputation (faster but less thorough)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.package_path, args.manifest, args.batch]):
        parser.error("Must specify package_path, --manifest, or --batch")
    
    verifier = PackageVerifier(verbose=args.verbose)
    
    try:
        if args.batch:
            # Batch verification
            report = verifier.verify_batch(args.batch)
            passed = report.get('packages_failed', 1) == 0
            
        elif args.manifest:
            # Single manifest verification
            manifest_path = Path(args.manifest)
            if not manifest_path.exists():
                print(f"Error: Manifest file not found: {manifest_path}")
                sys.exit(1)
            
            session_dir = manifest_path.parent
            passed, report = verifier._verify_manifest_in_directory(session_dir)
            verifier.results['packages_verified'] = 1
            if passed:
                verifier.results['packages_passed'] = 1
            else:
                verifier.results['packages_failed'] = 1
            
        else:
            # Single package verification
            passed, report = verifier.verify_package(args.package_path)
            verifier.results['packages_verified'] = 1
            if passed:
                verifier.results['packages_passed'] = 1
            else:
                verifier.results['packages_failed'] = 1
        
        # Save JSON report if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            verifier.log(f"Report saved to: {args.json}")
        
        # Print summary
        verifier.print_summary()
        
        # Print detailed report if not batch mode and verbose
        if args.verbose and not args.batch:
            print("\nDETAILED REPORT:")
            print(json.dumps(report, indent=2, default=str))
        
        # Exit with appropriate code
        sys.exit(0 if passed else 1)
        
    except KeyboardInterrupt:
        print("\nVerification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()