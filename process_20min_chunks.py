#!/usr/bin/env python3
"""
Process 20-minute chunks through AssemblyAI Ensemble Pipeline
Handles 5 chunks with 10 speakers configuration and generates synchronized outputs
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ensemble_manager import EnsembleManager
from utils.enhanced_structured_logger import create_enhanced_logger
from utils.transcript_formatter import TranscriptFormatter

# Configure logging
logger = create_enhanced_logger("20min_processor")

def main():
    """Process all 5 chunks through the ensemble pipeline"""
    
    # Configuration for 10 speakers (converts to 9 for AssemblyAI)
    expected_speakers = 10
    noise_level = 'medium'
    target_language = None  # Auto-detect
    domain = "general"
    
    # Scoring weights (optimized for accuracy)
    scoring_weights = {
        'D': 0.28,  # Diarization consistency
        'A': 0.32,  # ASR alignment and confidence  
        'L': 0.18,  # Linguistic quality
        'R': 0.12,  # Cross-run agreement
        'O': 0.10   # Overlap handling
    }
    
    # Speaker mapping configuration for 10 speakers
    speaker_mapping_config = {
        'similarity_threshold': 0.7,
        'embedding_dim': 128,
        'min_segment_duration': 1.0,
        'cache_embeddings': True,
        'enable_metrics': True
    }
    
    # Initialize ensemble manager
    logger.info("Initializing Ensemble Manager for 20-minute processing",
                expected_speakers=expected_speakers,
                chunk_count=5)
    
    ensemble = EnsembleManager(
        expected_speakers=expected_speakers,
        noise_level=noise_level,
        target_language=target_language,
        scoring_weights=scoring_weights,
        enable_versioning=True,
        domain=domain,
        consensus_strategy="best_single_candidate",
        calibration_method="registry_based",
        enable_speaker_mapping=True,
        speaker_mapping_config=speaker_mapping_config,
        chunked_processing_threshold=900.0  # 15 minutes threshold
    )
    
    # Chunk paths
    chunks_dir = Path("artifacts/20min_processing/chunks")
    chunk_files = sorted(chunks_dir.glob("chunk_*.mp4"))
    
    logger.info(f"Found {len(chunk_files)} chunks to process", 
                chunks=[f.name for f in chunk_files])
    
    # Results storage
    results_dir = Path("artifacts/20min_processing/results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = []
    processing_metadata = {
        "session_id": f"20min_processing_{int(time.time())}",
        "start_time": datetime.now().isoformat(),
        "configuration": {
            "expected_speakers": expected_speakers,
            "noise_level": noise_level,
            "domain": domain,
            "scoring_weights": scoring_weights,
            "consensus_strategy": "best_single_candidate",
            "calibration_method": "registry_based",
        },
        "chunks": [],
        "results": []
    }
    
    # Process each chunk
    for i, chunk_file in enumerate(chunk_files, 1):
        chunk_name = chunk_file.name
        logger.info(f"Processing chunk {i}/{len(chunk_files)}: {chunk_name}")
        
        try:
            # Process through ensemble
            start_time = time.time()
            
            result = ensemble.process_video(str(chunk_file))
            
            processing_time = time.time() - start_time
            
            logger.info(f"Chunk {i} processed successfully",
                       processing_time=processing_time,
                       chunk_name=chunk_name)
            
            # Save individual chunk result
            chunk_result_file = results_dir / f"chunk_{i:02d}_result.json"
            with open(chunk_result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Add to combined results
            chunk_metadata = {
                "chunk_index": i,
                "chunk_name": chunk_name,
                "chunk_file": str(chunk_file),
                "processing_time": processing_time,
                "result_file": str(chunk_result_file),
                "status": "completed"
            }
            processing_metadata["chunks"].append(chunk_metadata)
            all_results.append(result)
            
            logger.info(f"Chunk {i} results saved", result_file=str(chunk_result_file))
            
        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {chunk_name}",
                        error=str(e),
                        chunk_file=str(chunk_file))
            
            chunk_metadata = {
                "chunk_index": i,
                "chunk_name": chunk_name,
                "chunk_file": str(chunk_file),
                "processing_time": 0,
                "result_file": None,
                "status": "failed",
                "error": str(e)
            }
            processing_metadata["chunks"].append(chunk_metadata)
            
            # Continue with next chunk
            continue
    
    # Generate combined outputs
    processing_metadata["end_time"] = datetime.now().isoformat()
    processing_metadata["total_chunks"] = len(chunk_files)
    processing_metadata["successful_chunks"] = len([c for c in processing_metadata["chunks"] if c["status"] == "completed"])
    
    logger.info("All chunks processed",
                total=len(chunk_files),
                successful=processing_metadata["successful_chunks"])
    
    # Save processing metadata
    metadata_file = results_dir / "processing_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(processing_metadata, f, indent=2, ensure_ascii=False)
    
    # Generate synchronized outputs if we have successful results
    if all_results:
        try:
            # Create combined transcript formatter
            formatter = TranscriptFormatter()
            
            # Generate various output formats
            output_formats = ['json', 'srt', 'vtt', 'txt']
            
            for output_format in output_formats:
                combined_file = results_dir / f"combined_transcript.{output_format}"
                
                if output_format == 'json':
                    # JSON format with all metadata
                    combined_data = {
                        "metadata": processing_metadata,
                        "chunks": all_results
                    }
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, indent=2, ensure_ascii=False)
                        
                elif output_format == 'txt':
                    # Plain text transcript
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        for i, result in enumerate(all_results, 1):
                            f.write(f"=== CHUNK {i} ===\n")
                            if 'transcript' in result:
                                f.write(result['transcript'] + "\n\n")
                            elif 'best_transcript' in result:
                                f.write(result['best_transcript'] + "\n\n")
                
                logger.info(f"Generated {output_format.upper()} output", file=str(combined_file))
            
        except Exception as e:
            logger.error("Failed to generate combined outputs", error=str(e))
    
    # Summary
    logger.info("20-minute processing completed",
                metadata_file=str(metadata_file),
                results_dir=str(results_dir),
                total_chunks=len(chunk_files),
                successful=processing_metadata["successful_chunks"])
    
    print(f"\n✅ Processing completed!")
    print(f"📁 Results directory: {results_dir}")
    print(f"📄 Metadata file: {metadata_file}")
    print(f"✅ Successfully processed: {processing_metadata['successful_chunks']}/{len(chunk_files)} chunks")
    
    return processing_metadata

if __name__ == "__main__":
    main()