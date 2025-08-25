"""
CLI entry point for mesh optimization package
"""

import argparse
import sys
import logging
import json
from pathlib import Path

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mesh_optim.log')
        ]
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AortaCFD Mesh Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Geometry-driven mesh optimization
  python -m mesh_optim stage1 --geometry cases_input/patient1 --config mesh_optim/configs/stage1_default.json

  # Stage 2: Coming in future release
  # python -m mesh_optim stage2 --geometry cases_input/patient1 --model RANS
        """
    )
    
    subparsers = parser.add_subparsers(dest='stage', help='Optimization stage')
    
    # Stage 1 subparser
    stage1_parser = subparsers.add_parser('stage1', help='Geometry-driven mesh optimization')
    stage1_parser.add_argument('--geometry', required=True, help='Path to geometry directory (with STL files)')
    stage1_parser.add_argument('--config', help='Configuration file (default: stage1_default.json)')
    stage1_parser.add_argument('--output', help='Output directory (default: output/<patient>/meshOptimizer/stage1)')
    stage1_parser.add_argument('--max-iterations', type=int, default=4, help='Maximum iterations')
    stage1_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Stage 2 subparser - commented out until implementation is ready
    # stage2_parser = subparsers.add_parser('stage2', help='GCI-based WSS convergence verification')
    # stage2_parser.add_argument('--geometry', required=True, help='Path to geometry directory')
    # stage2_parser.add_argument('--model', choices=['LAMINAR', 'RANS', 'LES'], default='RANS', help='Flow model')
    
    args = parser.parse_args()
    
    if not args.stage:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('mesh_optim.cli')
    
    try:
        if args.stage == 'stage1':
            from .stage1_mesh import Stage1MeshOptimizer
            
            # Use default config if not specified
            if not args.config:
                args.config = Path(__file__).parent / "configs" / "stage1_default.json"
            
            # Generate default output path if not specified
            if not args.output:
                patient_name = Path(args.geometry).name
                args.output = Path("output") / patient_name / "meshOptimizer" / "stage1"
            
            logger.info(f"Starting Stage 1 optimization: {args.geometry}")
            logger.info(f"Output directory: {args.output}")
            optimizer = Stage1MeshOptimizer(args.geometry, args.config, args.output)
            
            if args.max_iterations:
                optimizer.max_iterations = args.max_iterations
                
            result_dir = optimizer.iterate_until_quality()
            logger.info(f"✅ Stage 1 completed: {result_dir}")
            
        elif args.stage == 'stage2':
            logger.error("❌ Stage 2 is not yet available in this release")
            logger.info("💡 Stage 2 (GCI verification) will be added in a future update")
            logger.info("💡 For now, use Stage 1 for high-quality mesh generation")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())