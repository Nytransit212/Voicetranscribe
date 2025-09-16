#!/usr/bin/env python3
"""
Cache Integrity Audit and Cleanup System

This script provides comprehensive cache management including:
- Integrity checking (missing sidecars, corrupted metadata, orphaned files)
- LRU eviction with configurable byte limits per project
- Stale cache cleanup based on age and validation
- Detailed reporting and metrics
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

# Add utils to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.cache_operations import AtomicCacheOperations, get_cache_operations
from utils.cache_key import CacheScope, CacheMetadata
from utils.enhanced_structured_logger import create_enhanced_logger


@dataclass
class CacheAuditStats:
    """Statistics from cache audit operations"""
    
    # File counts
    total_cache_entries: int = 0
    valid_entries: int = 0
    orphaned_data_files: int = 0
    orphaned_metadata_files: int = 0
    corrupted_metadata_files: int = 0
    missing_data_files: int = 0
    
    # Size information
    total_cache_bytes: int = 0
    bytes_by_scope: Dict[str, int] = field(default_factory=dict)
    bytes_by_project: Dict[str, int] = field(default_factory=dict)
    
    # Age information
    oldest_entry_days: float = 0.0
    newest_entry_days: float = 0.0
    avg_entry_age_days: float = 0.0
    
    # Cleanup statistics
    entries_cleaned: int = 0
    bytes_cleaned: int = 0
    lru_evictions: int = 0
    stale_evictions: int = 0
    corrupted_evictions: int = 0
    
    # Performance
    audit_duration_seconds: float = 0.0
    
    def add_cleanup(self, entries: int, bytes_cleaned: int, reason: str):
        """Add cleanup statistics"""
        self.entries_cleaned += entries
        self.bytes_cleaned += bytes_cleaned
        
        if reason == "lru":
            self.lru_evictions += entries
        elif reason == "stale":
            self.stale_evictions += entries
        elif reason == "corrupted":
            self.corrupted_evictions += entries


class CacheAuditor:
    """
    Comprehensive cache auditor with integrity checking and LRU eviction.
    
    Provides automated cache management to prevent storage bloat and
    ensure cache integrity across all processing stages.
    """
    
    def __init__(self, 
                 cache_ops: Optional[AtomicCacheOperations] = None,
                 soft_cap_gb: float = 8.0,
                 hard_cap_gb: float = 10.0,
                 max_age_days: int = 30):
        """
        Initialize cache auditor.
        
        Args:
            cache_ops: Cache operations instance (uses global if None)
            soft_cap_gb: Soft limit for cache size per project (triggers cleanup)
            hard_cap_gb: Hard limit for cache size per project (aggressive cleanup)
            max_age_days: Maximum age for cache entries before considered stale
        """
        self.cache_ops = cache_ops or get_cache_operations()
        self.soft_cap_bytes = int(soft_cap_gb * 1024**3)
        self.hard_cap_bytes = int(hard_cap_gb * 1024**3)
        self.max_age_days = max_age_days
        
        self.logger = create_enhanced_logger("cache_auditor")
        
        self.logger.info(
            "Cache auditor initialized",
            cache_directory=str(self.cache_ops.cache_base_dir),
            soft_cap_gb=soft_cap_gb,
            hard_cap_gb=hard_cap_gb,
            max_age_days=max_age_days
        )
    
    def scan_cache_integrity(self) -> CacheAuditStats:
        """
        Perform comprehensive cache integrity scan.
        
        Returns:
            CacheAuditStats with detailed scan results
        """
        start_time = time.time()
        stats = CacheAuditStats()
        
        self.logger.info("Starting cache integrity scan")
        
        try:
            # Scan all cache directories
            all_data_files = set()
            all_metadata_files = set()
            valid_entries = []
            
            # Find all cache files
            for cache_file in self.cache_ops.cache_base_dir.rglob("*"):
                if cache_file.is_file():
                    if cache_file.name.endswith(".data"):
                        all_data_files.add(cache_file)
                    elif cache_file.name.endswith(".meta.json"):
                        all_metadata_files.add(cache_file)
            
            self.logger.info(
                f"Found {len(all_data_files)} data files and {len(all_metadata_files)} metadata files"
            )
            
            # Check metadata files and their corresponding data files
            for metadata_path in all_metadata_files:
                expected_data_path = metadata_path.with_suffix('.data')
                
                # Load metadata
                metadata = self.cache_ops._load_metadata(metadata_path)
                if metadata is None:
                    stats.corrupted_metadata_files += 1
                    self.logger.warning(
                        f"Corrupted metadata file: {metadata_path}"
                    )
                    continue
                
                # Check if data file exists
                if expected_data_path not in all_data_files:
                    stats.missing_data_files += 1
                    self.logger.warning(
                        f"Missing data file for metadata: {metadata_path}"
                    )
                    continue
                
                # Valid entry
                stats.valid_entries += 1
                stats.total_cache_bytes += metadata.byte_size
                
                # Track by scope and project
                scope = self._extract_scope_from_path(metadata_path)
                project_id = self._extract_project_from_path(metadata_path)
                
                stats.bytes_by_scope[scope] = stats.bytes_by_scope.get(scope, 0) + metadata.byte_size
                if project_id:
                    stats.bytes_by_project[project_id] = stats.bytes_by_project.get(project_id, 0) + metadata.byte_size
                
                # Track age
                entry_age_days = self._calculate_age_days(metadata.created_at)
                if stats.oldest_entry_days == 0 or entry_age_days > stats.oldest_entry_days:
                    stats.oldest_entry_days = entry_age_days
                if stats.newest_entry_days == 0 or entry_age_days < stats.newest_entry_days:
                    stats.newest_entry_days = entry_age_days
                
                valid_entries.append((metadata_path, metadata, expected_data_path))
                all_data_files.discard(expected_data_path)  # Mark as paired
            
            # Count orphaned files
            stats.orphaned_data_files = len(all_data_files)
            stats.orphaned_metadata_files = len(all_metadata_files) - stats.valid_entries - stats.corrupted_metadata_files
            
            # Calculate average age
            if valid_entries:
                total_age = sum(self._calculate_age_days(metadata.created_at) for _, metadata, _ in valid_entries)
                stats.avg_entry_age_days = total_age / len(valid_entries)
            
            stats.total_cache_entries = len(all_metadata_files)
            stats.audit_duration_seconds = time.time() - start_time
            
            self.logger.info(
                "Cache integrity scan completed",
                total_entries=stats.total_cache_entries,
                valid_entries=stats.valid_entries,
                corrupted_metadata=stats.corrupted_metadata_files,
                orphaned_data=stats.orphaned_data_files,
                orphaned_metadata=stats.orphaned_metadata_files,
                missing_data=stats.missing_data_files,
                total_cache_gb=round(stats.total_cache_bytes / 1024**3, 2),
                audit_duration=round(stats.audit_duration_seconds, 2)
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Cache integrity scan failed: {e}")
            stats.audit_duration_seconds = time.time() - start_time
            return stats
    
    def cleanup_corrupted_entries(self) -> CacheAuditStats:
        """
        Clean up corrupted and orphaned cache entries.
        
        Returns:
            CacheAuditStats with cleanup results
        """
        stats = CacheAuditStats()
        
        self.logger.info("Starting corrupted entry cleanup")
        
        try:
            # Find all files
            all_data_files = set()
            all_metadata_files = set()
            
            for cache_file in self.cache_ops.cache_base_dir.rglob("*"):
                if cache_file.is_file():
                    if cache_file.name.endswith(".data"):
                        all_data_files.add(cache_file)
                    elif cache_file.name.endswith(".meta.json"):
                        all_metadata_files.add(cache_file)
            
            cleanup_files = []
            cleanup_bytes = 0
            
            # Check metadata files
            for metadata_path in all_metadata_files:
                expected_data_path = metadata_path.with_suffix('.data')
                
                # Try to load metadata
                metadata = self.cache_ops._load_metadata(metadata_path)
                if metadata is None:
                    # Corrupted metadata
                    cleanup_files.extend([metadata_path, expected_data_path])
                    if expected_data_path.exists():
                        cleanup_bytes += expected_data_path.stat().st_size
                    continue
                
                # Check if data file exists
                if not expected_data_path.exists():
                    # Missing data file - remove metadata
                    cleanup_files.append(metadata_path)
                    continue
                
                # Remove from orphaned set if valid
                all_data_files.discard(expected_data_path)
            
            # Add orphaned data files
            for orphaned_data in all_data_files:
                cleanup_files.append(orphaned_data)
                cleanup_bytes += orphaned_data.stat().st_size
            
            # Perform cleanup
            for file_path in cleanup_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        stats.entries_cleaned += 1
                        
                        self.logger.debug(f"Removed corrupted/orphaned file: {file_path}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to remove file {file_path}: {e}")
            
            stats.bytes_cleaned = cleanup_bytes
            stats.corrupted_evictions = stats.entries_cleaned
            
            self.logger.info(
                "Corrupted entry cleanup completed",
                files_removed=stats.entries_cleaned,
                bytes_cleaned_mb=round(cleanup_bytes / 1024**2, 2)
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Corrupted entry cleanup failed: {e}")
            return stats
    
    def cleanup_stale_entries(self, max_age_days: Optional[int] = None) -> CacheAuditStats:
        """
        Clean up stale cache entries based on age.
        
        Args:
            max_age_days: Maximum age in days (uses default if None)
            
        Returns:
            CacheAuditStats with cleanup results
        """
        max_age = max_age_days or self.max_age_days
        stats = CacheAuditStats()
        
        self.logger.info(f"Starting stale entry cleanup (max age: {max_age} days)")
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age)
            cleanup_files = []
            cleanup_bytes = 0
            
            # Scan all metadata files
            for metadata_path in self.cache_ops.cache_base_dir.rglob("*.meta.json"):
                metadata = self.cache_ops._load_metadata(metadata_path)
                if metadata is None:
                    continue
                
                # Check age
                try:
                    created_time = datetime.fromisoformat(metadata.created_at.replace('Z', '+00:00'))
                    if created_time < cutoff_time:
                        # Stale entry
                        data_path = metadata_path.with_suffix('.data')
                        cleanup_files.extend([metadata_path, data_path])
                        cleanup_bytes += metadata.byte_size
                        
                except (ValueError, TypeError):
                    # Invalid timestamp - treat as stale
                    data_path = metadata_path.with_suffix('.data')
                    cleanup_files.extend([metadata_path, data_path])
                    cleanup_bytes += metadata.byte_size
            
            # Perform cleanup
            for file_path in cleanup_files:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        stats.entries_cleaned += 1
                        
                        self.logger.debug(f"Removed stale file: {file_path}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to remove stale file {file_path}: {e}")
            
            stats.bytes_cleaned = cleanup_bytes
            stats.stale_evictions = stats.entries_cleaned // 2  # Each entry has 2 files
            
            self.logger.info(
                "Stale entry cleanup completed",
                entries_removed=stats.stale_evictions,
                files_removed=stats.entries_cleaned,
                bytes_cleaned_mb=round(cleanup_bytes / 1024**2, 2)
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Stale entry cleanup failed: {e}")
            return stats
    
    def lru_eviction_by_project(self, target_bytes: Optional[int] = None) -> CacheAuditStats:
        """
        Perform LRU eviction per project to stay under byte limits.
        
        Args:
            target_bytes: Target cache size per project (uses soft_cap if None)
            
        Returns:
            CacheAuditStats with eviction results
        """
        target = target_bytes or self.soft_cap_bytes
        stats = CacheAuditStats()
        
        self.logger.info(f"Starting LRU eviction (target: {target / 1024**3:.2f} GB per project)")
        
        try:
            # Group entries by project
            project_entries = defaultdict(list)
            
            # Scan all entries
            for metadata_path in self.cache_ops.cache_base_dir.rglob("*.meta.json"):
                metadata = self.cache_ops._load_metadata(metadata_path)
                if metadata is None:
                    continue
                
                project_id = self._extract_project_from_path(metadata_path)
                if not project_id:
                    project_id = "global"
                
                # Add last access time for LRU sorting
                last_access = metadata.last_accessed_at or metadata.created_at
                try:
                    access_time = datetime.fromisoformat(last_access.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    access_time = datetime.min.replace(tzinfo=timezone.utc)
                
                project_entries[project_id].append({
                    'metadata_path': metadata_path,
                    'data_path': metadata_path.with_suffix('.data'),
                    'metadata': metadata,
                    'access_time': access_time,
                    'size': metadata.byte_size
                })
            
            # Process each project
            total_cleanup_bytes = 0
            total_cleanup_entries = 0
            
            for project_id, entries in project_entries.items():
                project_size = sum(entry['size'] for entry in entries)
                
                if project_size <= target:
                    # Project under limit
                    continue
                
                self.logger.info(
                    f"Project '{project_id}' over limit",
                    current_size_gb=round(project_size / 1024**3, 2),
                    target_size_gb=round(target / 1024**3, 2),
                    entries_count=len(entries)
                )
                
                # Sort by access time (oldest first for LRU)
                entries.sort(key=lambda x: x['access_time'])
                
                # Remove entries until under target
                cleanup_size = 0
                cleanup_count = 0
                
                for entry in entries:
                    if project_size - cleanup_size <= target:
                        break
                    
                    # Remove this entry
                    try:
                        for file_path in [entry['metadata_path'], entry['data_path']]:
                            if file_path.exists():
                                file_path.unlink()
                        
                        cleanup_size += entry['size']
                        cleanup_count += 1
                        
                        self.logger.debug(
                            f"LRU evicted entry",
                            project_id=project_id,
                            cache_key_short=entry['metadata'].cache_key_short,
                            size_mb=round(entry['size'] / 1024**2, 2)
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Failed to remove LRU entry: {e}")
                
                total_cleanup_bytes += cleanup_size
                total_cleanup_entries += cleanup_count
                
                self.logger.info(
                    f"LRU eviction completed for project '{project_id}'",
                    entries_removed=cleanup_count,
                    bytes_removed_gb=round(cleanup_size / 1024**3, 2),
                    final_size_gb=round((project_size - cleanup_size) / 1024**3, 2)
                )
            
            stats.bytes_cleaned = total_cleanup_bytes
            stats.lru_evictions = total_cleanup_entries
            stats.entries_cleaned = total_cleanup_entries
            
            self.logger.info(
                "LRU eviction completed",
                total_entries_removed=total_cleanup_entries,
                total_bytes_removed_gb=round(total_cleanup_bytes / 1024**3, 2)
            )
            
            return stats
            
        except Exception as e:
            self.logger.error(f"LRU eviction failed: {e}")
            return stats
    
    def full_audit_and_cleanup(self, 
                              cleanup_corrupted: bool = True,
                              cleanup_stale: bool = True,
                              lru_eviction: bool = True) -> CacheAuditStats:
        """
        Perform comprehensive audit and cleanup operations.
        
        Args:
            cleanup_corrupted: Whether to clean up corrupted entries
            cleanup_stale: Whether to clean up stale entries
            lru_eviction: Whether to perform LRU eviction
            
        Returns:
            Combined CacheAuditStats from all operations
        """
        start_time = time.time()
        
        self.logger.info("Starting full cache audit and cleanup")
        
        # Initial integrity scan
        integrity_stats = self.scan_cache_integrity()
        
        # Combined stats
        combined_stats = integrity_stats
        
        # Cleanup corrupted entries
        if cleanup_corrupted and (integrity_stats.corrupted_metadata_files > 0 or 
                                 integrity_stats.orphaned_data_files > 0 or
                                 integrity_stats.missing_data_files > 0):
            self.logger.info("Running corrupted entry cleanup")
            corrupted_stats = self.cleanup_corrupted_entries()
            combined_stats.add_cleanup(
                corrupted_stats.entries_cleaned,
                corrupted_stats.bytes_cleaned,
                "corrupted"
            )
        
        # Cleanup stale entries
        if cleanup_stale:
            self.logger.info("Running stale entry cleanup")
            stale_stats = self.cleanup_stale_entries()
            combined_stats.add_cleanup(
                stale_stats.stale_evictions,
                stale_stats.bytes_cleaned,
                "stale"
            )
        
        # LRU eviction
        if lru_eviction:
            # Check if any project is over soft cap
            over_limit_projects = [
                project for project, size in combined_stats.bytes_by_project.items()
                if size > self.soft_cap_bytes
            ]
            
            if over_limit_projects:
                self.logger.info(f"Running LRU eviction for {len(over_limit_projects)} projects over limit")
                lru_stats = self.lru_eviction_by_project()
                combined_stats.add_cleanup(
                    lru_stats.lru_evictions,
                    lru_stats.bytes_cleaned,
                    "lru"
                )
        
        combined_stats.audit_duration_seconds = time.time() - start_time
        
        self.logger.info(
            "Full cache audit and cleanup completed",
            total_duration=round(combined_stats.audit_duration_seconds, 2),
            total_entries_cleaned=combined_stats.entries_cleaned,
            total_bytes_cleaned_gb=round(combined_stats.bytes_cleaned / 1024**3, 2),
            lru_evictions=combined_stats.lru_evictions,
            stale_evictions=combined_stats.stale_evictions,
            corrupted_evictions=combined_stats.corrupted_evictions
        )
        
        return combined_stats
    
    def _extract_scope_from_path(self, path: Path) -> str:
        """Extract cache scope from file path"""
        path_parts = path.parts
        for part in path_parts:
            if part in ("global", "projects", "sessions", "runs"):
                return part
        return "unknown"
    
    def _extract_project_from_path(self, path: Path) -> Optional[str]:
        """Extract project ID from file path"""
        path_parts = path.parts
        try:
            projects_idx = path_parts.index("projects")
            if projects_idx + 1 < len(path_parts):
                return path_parts[projects_idx + 1]
        except ValueError:
            pass
        return None
    
    def _calculate_age_days(self, timestamp_str: str) -> float:
        """Calculate age in days from ISO timestamp"""
        try:
            created_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age = datetime.now(timezone.utc) - created_time
            return age.total_seconds() / (24 * 3600)
        except (ValueError, TypeError):
            return 0.0


def main():
    """Main CLI interface for cache auditing"""
    parser = argparse.ArgumentParser(description="Cache Audit and Cleanup Tool")
    
    parser.add_argument("--action", 
                       choices=["scan", "cleanup-corrupted", "cleanup-stale", "lru-evict", "full"],
                       default="scan",
                       help="Action to perform")
    
    parser.add_argument("--cache-dir", 
                       type=str,
                       help="Cache directory path")
    
    parser.add_argument("--soft-cap-gb", 
                       type=float, 
                       default=8.0,
                       help="Soft cache size limit per project (GB)")
    
    parser.add_argument("--hard-cap-gb", 
                       type=float, 
                       default=10.0,
                       help="Hard cache size limit per project (GB)")
    
    parser.add_argument("--max-age-days", 
                       type=int, 
                       default=30,
                       help="Maximum age for cache entries (days)")
    
    parser.add_argument("--output-format",
                       choices=["text", "json"],
                       default="text",
                       help="Output format for results")
    
    parser.add_argument("--verbose", "-v",
                       action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = create_enhanced_logger("cache_audit_cli")
    
    # Initialize cache operations
    cache_ops = None
    if args.cache_dir:
        cache_ops = AtomicCacheOperations(args.cache_dir)
    
    # Initialize auditor
    auditor = CacheAuditor(
        cache_ops=cache_ops,
        soft_cap_gb=args.soft_cap_gb,
        hard_cap_gb=args.hard_cap_gb,
        max_age_days=args.max_age_days
    )
    
    # Perform requested action
    if args.action == "scan":
        stats = auditor.scan_cache_integrity()
    elif args.action == "cleanup-corrupted":
        stats = auditor.cleanup_corrupted_entries()
    elif args.action == "cleanup-stale":
        stats = auditor.cleanup_stale_entries()
    elif args.action == "lru-evict":
        stats = auditor.lru_eviction_by_project()
    elif args.action == "full":
        stats = auditor.full_audit_and_cleanup()
    else:
        logger.error(f"Unknown action: {args.action}")
        sys.exit(1)
    
    # Output results
    if args.output_format == "json":
        import json
        from dataclasses import asdict
        print(json.dumps(asdict(stats), indent=2))
    else:
        # Text format
        print(f"Cache Audit Results ({args.action})")
        print("=" * 50)
        print(f"Total cache entries: {stats.total_cache_entries}")
        print(f"Valid entries: {stats.valid_entries}")
        print(f"Total cache size: {stats.total_cache_bytes / 1024**3:.2f} GB")
        
        if stats.entries_cleaned > 0:
            print(f"Entries cleaned: {stats.entries_cleaned}")
            print(f"Bytes cleaned: {stats.bytes_cleaned / 1024**3:.2f} GB")
            print(f"LRU evictions: {stats.lru_evictions}")
            print(f"Stale evictions: {stats.stale_evictions}")
            print(f"Corrupted evictions: {stats.corrupted_evictions}")
        
        print(f"Audit duration: {stats.audit_duration_seconds:.2f} seconds")
        
        if stats.bytes_by_project:
            print("\nCache size by project:")
            for project, size in sorted(stats.bytes_by_project.items()):
                print(f"  {project}: {size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()